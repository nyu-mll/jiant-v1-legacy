"""
<<<<<<< HEAD
Functions to do with parsing configs here.
"""
import torch
=======
Functions for parsing configs.
"""
import torch
import logging as log
>>>>>>> master


def parse_task_list_arg(task_list):
    """Parse task list argument into a list of task names."""
    task_names = []
    for task_name in task_list.split(","):
        if task_name == "glue":
            task_names.extend(ALL_GLUE_TASKS)
        elif task_name == "superglue":
            task_names.extend(ALL_SUPERGLUE_TASKS)
        elif task_name == "none" or task_name == "":
            continue
        else:
            task_names.append(task_name)
    return task_names


<<<<<<< HEAD
def parse_cuda_list_arg(cuda_list):
=======
def parse_cuda_list_arg(cuda_arg):
>>>>>>> master
    """
    Parse cuda_list settings
    """
    result_cuda = []
<<<<<<< HEAD
    if cuda_list == "auto":
        cuda_list = list(range(torch.cuda.device_count()))
        return cuda_list
    if isinstance(cuda_list, int):
        result_cuda = [cuda_list]
    elif "," in cuda_list:
        for device_id in cuda_list.split(","):
            result_cuda.append(int(device_id))
=======
    if cuda_arg == "auto":
        result_cuda = list(range(torch.cuda.device_count()))
        if len(result_cuda) == 1:
            result_cuda = result_cuda[0]
        elif len(result_cuda) == 0:
            result_cuda = -1
    elif isinstance(cuda_arg, int):
        result_cuda = cuda_arg
    elif "," in cuda_arg:
        result_cuda = [int(d) for d in cuda_arg.split(",")]
>>>>>>> master
    else:
        raise ValueError(
            "Your cuda settings do not match any of the possibilities in defaults.conf"
        )
<<<<<<< HEAD
    if len(result_cuda) == 1:
        result_cuda = result_cuda[0]
=======
    if torch.cuda.device_count() == 0 and not (isinstance(result_cuda, int) and result_cuda == -1):
        raise ValueError("You specified usage of CUDA but CUDA devices not found.")
>>>>>>> master
    return result_cuda
