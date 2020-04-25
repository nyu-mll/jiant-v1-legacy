import os
import json
import numpy

JIANT_PROJECT_PREFIX = os.getenv("JIANT_PROJECT_PREFIX")
JIANT_DATA_DIR = os.getenv("JIANT_DATA_DIR")
RANDOM_SEEDS = [432, 5287, 98235]
DEVICE = "p100"
mixing_K = 16384
metadata_file = os.path.join(os.path.dirname(__file__), "task_metadata.json")
num_main_trials = 10
num_addi_trials = 3


def load_metadata():
    with open(metadata_file, "r") as f:
        task_metadata = json.loads(f.read())
    return task_metadata


def save_metadata(task_metadata):
    with open(metadata_file, "w") as f:
        f.write(json.dumps(task_metadata))


def get_batch_size_limit(task_info, input_module):
    batch_size_limit = task_info[f'{input_module.split("-")[0]}_batch_size_limit']
    if isinstance(batch_size_limit, int):
        return batch_size_limit
    else:
        return batch_size_limit[DEVICE]


def batch_size_limit_to_gpus(batch_size_limit, jiant):
    if batch_size_limit <= 4:
        gpu_available, sbatch = 4, ("jiant_gpu4.sbatch" if jiant else "python_gpu4.sbatch")
    elif batch_size_limit == 8:
        gpu_available, sbatch = 2, ("jiant_gpu2.sbatch" if jiant else "python_gpu2.sbatch")
    else:
        gpu_available, sbatch = 1, ("jiant_gpu1.sbatch" if jiant else "python_gpu1.sbatch")
    sbatch = os.path.join("scripts/taskmaster/gcp", sbatch)
    return gpu_available, sbatch


cpu_sbatch = os.path.join("scripts/taskmaster/gcp", "cpu.sbatch")
basic_jiant_sbatch = batch_size_limit_to_gpus(numpy.inf, jiant=True)[1]


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
