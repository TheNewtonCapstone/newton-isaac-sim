from typing import List, Optional

from GPUtil import GPUtil, GPU


def get_gpus() -> List[GPU]:
    gpus = GPUtil.getGPUs()
    assert len(gpus) > 0, "No GPU devices found"

    return gpus


def get_free_gpu_memory(main_gpu: Optional[GPU] = None) -> int:
    if not main_gpu:
        main_gpu = get_gpus()[0]

    return main_gpu.memoryFree
