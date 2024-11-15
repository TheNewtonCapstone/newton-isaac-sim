import numpy as np
import torch
from core.agents.base_agent import BaseAgent
from core.base.base_env import BaseEnv
from core.domain_randomizer.domain_randomizer import DomainRandomizer
from core.newton.newton_agent import NewtonAgent
from core.sensors.imu.imu import IMU


# TODO: should be called GenericTwipEnv
class GenericEnv(BaseEnv):
    def __init__(
        self, world_settings, num_envs, terrain_builders, randomization_settings
    ):
        super().__init__(
            world_settings, num_envs, terrain_builders, randomization_settings
        )

        self.base_agent_path: str = ""

        self.agents_art_view = None

    def construct(self, agent: NewtonAgent) -> bool:
        super().construct(agent)

        from omni.isaac.cloner import GridCloner
        from omni.isaac.core.articulations import ArticulationView
        from omni.isaac.core.utils.stage import get_current_stage
        from omni.isaac.core.utils.prims import define_prim

        # add a ground plane
        self.terrain_builders[0].build_from_self(get_current_stage(), [0, 0, 0])

        # clone the agent
        cloner = GridCloner(spacing=1)
        cloner.define_base_env("/World/envs")
        self.base_agent_path = "/World/envs/e_0"
        define_prim(self.base_agent_path)

        self.agent.construct(self.base_agent_path, self.world)

        self.agent_paths = cloner.generate_paths("/World/envs/e", self.num_envs)
        self.agent_imu_paths = [f"{path}/newton/BASE/imu" for path in self.agent_paths]

        cloner.filter_collisions(
            physicsscene_path="/physicsScene",
            collision_root_path="/collisionGroups",
            prim_paths=self.agent_paths,
            global_paths=["/World/groundPlane"],
        )
        cloner.clone(
            source_prim_path=self.base_agent_path,
            prim_paths=self.agent_paths,
        )

        self.agents_art_view = ArticulationView(
            prim_paths_expr="/World/envs/e.*/newton/base",
            name="newton_art_view",
        )

        self.world.scene.add(self.agents_art_view)

        self.world.reset()

        self.imu = IMU(
            {
                "prim_path": "/World/envs/e.*/newton/base",
                "history_length": 0,
                "update_period": 0,
                "offset": {"pos": (0, 0, 0), "rot": (1.0, 0.0, 0.0, 0.0)},
            }
        )

        if self.randomize:
            self.domain_randomizer = DomainRandomizer(
                self.world,
                self.num_envs,
                self.agents_art_view,
                self.randomization_params,
            )
            print("Domain randomizer initialized")
            self.domain_randomizer.apply_randomization()

        return self.base_agent_path

    def step(self, actions: torch.Tensor, render: bool) -> torch.Tensor:
        if self.randomize:
            self.domain_randomizer.step_randomization()

        self._apply_actions(actions)

        if not render:
            self.world.app.update()

        self.world.step(render=render)

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

        self.agents_art_view.set_joint_velocity_targets(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        # using set_velocities instead of individual methods (lin & ang), because it's the only method supported in the GPU pipeline
        self.agents_art_view.set_velocities(
            torch.zeros(num_to_reset, 6), indices=indices
        )
        self.agents_art_view.set_joint_positions(
            torch.zeros(num_to_reset, 2), indices=indices
        )
        self.agents_art_view.set_joint_efforts(
            torch.zeros(num_to_reset, 2), indices=indices
        )

        # orientation at rest for the agents
        orientations = torch.tile(
            torch.tensor([0.98037, -0.18795, -0.01142, 0.05846]), (num_to_reset, 1)
        )

        # from GridCloner
        # translations should arrange all agents in a grid, with a spacing of 1, even if it's not a perfect square
        # an agent should always be at the same position in the grid (same index as specified in indices)
        spacing = 1
        num_per_row = int(np.sqrt(self.num_envs))
        num_rows = np.ceil(self.num_envs / num_per_row)
        num_cols = np.ceil(self.num_envs / num_rows)

        row_offset = 0.5 * spacing * (num_rows - 1)
        col_offset = 0.5 * spacing * (num_cols - 1)

        translations = torch.zeros(num_to_reset, 3)

        for i, idx in enumerate(indices):
            row = idx // num_cols
            col = idx % num_cols
            x = row_offset - row * spacing
            y = col * spacing - col_offset

            translations[i, 0] = x
            translations[i, 1] = y
            translations[i, 2] = 0.115

        self.agents_art_view.set_local_poses(
            translations=translations,
            orientations=orientations,
            indices=indices,
        )

        return

    def _apply_actions(self, torques: torch.Tensor) -> None:
        self.agents_art_view.set_joint_efforts(torques)

    def _gather_imus_frame(self) -> torch.Tensor:
        imu_data = self.imu.data

        # to(cpu) because the simulation may run on GPU and we want to eventually pass this data to STB3
        # which uses numpy arrays (on CPU)
        return torch.cat(
            (imu_data.lin_acc_b, imu_data.ang_vel_b, imu_data.quat_w), dim=1
        ).to(device="cpu")
