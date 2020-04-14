import os
import json
import numpy

JIANT_PROJECT_PREFIX = os.getenv("JIANT_PROJECT_PREFIX")
JIANT_DATA_DIR = os.getenv("JIANT_DATA_DIR")
RANDOM_SEEDS = [432, 5287, 98235, 8915, 2894]
metadata_file = os.path.join(os.path.dirname(__file__), "task_metadata.json")


def load_metadata():
    with open(metadata_file, "r") as f:
        task_metadata = json.loads(f.read())
    return task_metadata


def save_metadata(task_metadata):
    with open(metadata_file, "w") as f:
        f.write(json.dumps(task_metadata))


# TODO: rename 4p40, 2p40, p40 after prince jobs are all finished
def batch_size_limit_to_gpus(batch_size_limit, jiant):
    if batch_size_limit <= 4:
        gpu_available, sbatch = 4, ("jiant_gpu1.sbatch" if jiant else "4p40.sbatch")
    elif batch_size_limit == 8:
        gpu_available, sbatch = 2, ("jiant_gpu2.sbatch" if jiant else "2p40.sbatch")
    else:
        gpu_available, sbatch = 1, ("jiant_gpu4.sbatch" if jiant else "p40.sbatch")
    sbatch = os.path.join("scripts/taskmaster/gcp", sbatch)
    return gpu_available, sbatch


cpu_sbatch = os.path.join("scripts/taskmaster/gcp", "cpu.sbatch")
basic_jiant_sbatch = batch_size_limit_to_gpus(numpy.inf, jiant=True)


def batch_size_to_accumulation(batch_size_limit, batch_size, gpu_available):
    gpu_needed = batch_size // batch_size_limit
    if gpu_needed <= gpu_available:
        real_batch_size = batch_size
        accumulation_steps = 1
    else:
        assert gpu_needed % gpu_available == 0
        accumulation_steps = gpu_needed // gpu_available
        assert batch_size % accumulation_steps == 0
        real_batch_size = batch_size // accumulation_steps

    return real_batch_size, accumulation_steps
