import math

import torch
from core.base.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.domain_randomizer.domain_randomizer import DomainRandomizer
from core.sensors.imu.imu import IMU


class ProceduralEnv(BaseEnv):
    def __init__(
        self, world_settings, num_envs, terrain_builders, randomization_settings
    ) -> None:
        super().__init__(
            world_settings, num_envs, terrain_builders, randomization_settings
        )

        self.agent_positions = torch.zeros(self.num_envs, 3)
        self.randomize = randomization_settings.get("randomize", False)
        self.randomization_params = randomization_settings.get(
            "randomization_params", {}
        )

    def construct(self, agent: BaseAgent) -> bool:
        super().construct(agent)

        from omni.isaac.cloner import Cloner
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.utils.prims import define_prim

        stage = get_current_stage()
        num_terrains = len(self.terrain_builders)
        terrains_size = self.terrain_builders[0].size

        # generates a list of positions for each of the terrains, in a grid pattern
        perf_num_terrains_side = math.ceil(math.sqrt(num_terrains))
        terrain_positions = torch.tensor(
            [
                [
                    (i % perf_num_terrains_side) * terrains_size[0]
                    - terrains_size[0] / 2,
                    (i // perf_num_terrains_side) * terrains_size[1]
                    - terrains_size[1] / 2,
                    0,
                ]
                for i in range(num_terrains)
            ]
        ).tolist()

        agent_batch_qty = int(math.ceil(self.num_envs / num_terrains))

        from core.utils.physics import raycast

        # build & add all given terrains
        for i, terrain_builder in enumerate(self.terrain_builders):
            terrain_spawn_position = terrain_positions[i]

            assert (
                terrain_builder.size == terrains_size
            ), "All terrains must have the same size"

            terrain = terrain_builder.build_from_self(stage, terrain_spawn_position)

            self.terrain_paths.append(terrain.path)

            # propagate physics changes
            self.world.reset()

            # from the raycast, we can get the desired position of the agent to avoid clipping with the terrain
            raycast_height = 5
            max_ray_test_dist = 100
            min_ray_dist = max_ray_test_dist
            num_rays = 9
            rays_side = math.isqrt(num_rays)
            ray_separation = 0.1

            for j in range(num_rays):
                # we also want to cover a grid of rays on the xy-plane
                start_x = -ray_separation * (rays_side / 2)
                start_y = -ray_separation * (rays_side / 2)
                ray_x = ray_separation * (j % rays_side) + start_x
                ray_y = ray_separation * (j // rays_side) + start_y

                _, _, dist = raycast(
                    [
                        terrain_spawn_position[0] + ray_x,
                        terrain_spawn_position[1] + ray_y,
                        raycast_height,
                    ],
                    [0, 0, -1],
                    max_distance=max_ray_test_dist,
                )

                min_ray_dist = min(dist, min_ray_dist)

            # we want all agents to be evenly split across all terrains
            agent_batch_start = i * agent_batch_qty
            agent_batch_end = i * agent_batch_qty + agent_batch_qty

            self.agent_positions[agent_batch_start:agent_batch_end, :] = torch.tensor(
                # TODO: make it dependent on the agent's contact point
                [
                    terrain_spawn_position[0],
                    terrain_spawn_position[1],
                    raycast_height - min_ray_dist + 0.115,
                ]
            )

        # in some cases, ceil will give us more positions than we need
        if len(self.agent_positions) > self.num_envs:
            self.agent_positions = self.agent_positions[: self.num_envs]

        # clone the agent
        cloner = Cloner()
        cloner.define_base_env("/World/envs")
        base_agent_path = "/World/envs/e_0"
        define_prim(base_agent_path)

        self.agent.construct(base_agent_path, self.world)

        agent_paths = cloner.generate_paths("/World/envs/e", self.num_envs)

        cloner.filter_collisions(
            physicsscene_path="/physicsScene",
            collision_root_path="/collisionGroups",
            prim_paths=agent_paths,
            global_paths=["/World/groundPlane"] + self.terrain_paths,
        )
        cloner.clone(
            source_prim_path=base_agent_path,
            prim_paths=agent_paths,
            positions=self.agent_positions,
        )

        self.twip_art_view = ArticulationView(
            prim_paths_expr="/World/envs/e.*/twip/body",
            name="twip_art_view",
        )
        self.world.scene.add(self.twip_art_view)

        self.world.reset()

        self.imu = IMU(
            {
                "prim_path": "/World/envs/e.*/twip/body",
                "history_length": 0,
                "update_period": 0,
                "offset": {"pos": (0, 0, 0), "rot": (1.0, 0.0, 0.0, 0.0)},
            }
        )

        if self.randomize:
            self.domain_randomizer = DomainRandomizer(
                self.world, self.num_envs, self.twip_art_view, self.randomization_params
            )
            print("Domain randomizer initialized")
            self.domain_randomizer.apply_randomization()

        return base_agent_path

    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        if self.randomize:
            self.domain_randomizer.step_randomization()

            def randomize_terrain_properties():
                from core.utils.physics import set_physics_properties
                from random import Random

                print("Randomizing terrain properties")

                rnd = Random()
                for terrain_path in self.terrain_paths:
                    set_physics_properties(
                        terrain_path,
                        dynamic_friction=rnd.uniform(0.6, 1.4),
                        static_friction=rnd.uniform(0.6, 1.4),
                        restitution=rnd.uniform(0.0, 0.1),
                    )

            if self.domain_randomizer.frame_idx % self.domain_randomizer.frequency == 0:
                randomize_terrain_properties()

        # apply actions to the cloned agents
        self._apply_actions(actions)

        # From IsaacLab (SimulationContext)
        # need to do one step to refresh the app
        # reason: physics has to parse the scene again and inform other extensions like hydra-delegate.
        # without this the app becomes unresponsive. If render is True, the world updates the app automatically.
        if not render:
            self.world.app.update()

        self.world.step(render=render)

        # get observations from the cloned agents
        self.imu.update(self.world.get_physics_dt())
        obs = self._gather_imus_frame()

        return obs

    def reset(
        self,
        indices: torch.LongTensor | None = None,
    ) -> None:
        assert indices is None or indices.ndim == 1, "Indices must be a 1D tensor"

        # we assume it's a full reset
        if indices is None:
            self.world.reset()  # reset the world too, because we're doing a full reset

            indices = torch.arange(self.num_envs)

        num_to_reset = len(indices)

        self.twip_art_view.set_joint_velocity_targets(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        # using set_velocities instead of individual methods (lin & ang),
        # because it's the only method supported in the GPU pipeline
        self.twip_art_view.set_velocities(torch.zeros(num_to_reset, 6), indices=indices)
        self.twip_art_view.set_joint_positions(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        self.twip_art_view.set_joint_efforts(
            torch.zeros(num_to_reset, 2), indices=indices
        )

        from gymnasium.spaces import Box

        # generate a tensor with random values for the agents' roll & yaw within the specified range
        rand_yaws = Box(low=-math.pi, high=math.pi, shape=(num_to_reset,)).sample()
        max_roll = 0.35
        rand_rolls = Box(low=-max_roll, high=max_roll, shape=(num_to_reset,)).sample()

        from omni.isaac.core.utils.torch import quat_from_euler_xyz

        # convert the euler angles to quaternions
        orientations = quat_from_euler_xyz(
            torch.from_numpy(rand_rolls),
            torch.zeros((num_to_reset,)),
            torch.from_numpy(rand_yaws),
            extrinsic=True,
        )

        # ensure that we're on the same device (since we don't know which one in advance)
        if self.agent_positions.device != indices.device:
            self.agent_positions = self.agent_positions.to(indices.device)

        translations = self.agent_positions[indices]

        self.twip_art_view.set_local_poses(
            translations=translations,
            orientations=orientations,
            indices=indices,
        )

        return

    def _apply_actions(self, torques: torch.Tensor) -> None:
        self.twip_art_view.set_joint_efforts(torques)

    def _gather_imus_frame(self) -> torch.Tensor:
        imu_data = self.imu.data
        return torch.cat(
            (imu_data.lin_acc_b, imu_data.ang_vel_b, imu_data.quat_w), dim=1
        ).to(device="cpu")
