import logging
import torch
import numpy as np
import random
import pynvml # NVIDIA graphics card management Library
import os
import datetime

logger = logging.getLogger('MyMSA')

def setup_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed) # random seed for numpy
    random.seed(seed) # random seed for random module
    torch.manual_seed(seed) # random seed for CPU
    torch.backends.cudnn.enabled = False # Forbid CUDNN from using undeterministic algorithm
    torch.backends.cudnn.benchmark = False # benchmark mode (True) can increase computing speed while having randomness
    torch.backends.cudnn.deterministic = True # avoid the randomness in the calculation for feedward network
    torch.cuda.manual_seed(seed) # random seed for GPU
    torch.cuda.manual_seed_all(seed) # random seed for all GPUs

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def assign_gpu(args, gpu_ids, memory_limit=1e16):
    if len(gpu_ids) <= 1: # Single GPU Training
        args.distributed = False
        if len(gpu_ids) == 0 and torch.cuda.is_available(): # Find most free GPU
            pynvml.nvmlInit()
            n_gpus = pynvml.nvmlDeviceGetCount() # Number of GPU
            dst_gpu_id, min_mem_used = 0, memory_limit
            for g_id in range(n_gpus):
                handle = pynvml.nvmlDeviceGetHandleByIndex(g_id)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                mem_used = meminfo.used
                if mem_used < min_mem_used:
                    min_mem_used = mem_used
                    dst_gpu_id = g_id
            logger.info(f"Found most free GPU: ID {dst_gpu_id}, used memory {min_mem_used/(1024 * 1024)} MB")
            gpu_ids.append(dst_gpu_id)
        # Training device GPU or CPU
        using_cuda  = len(gpu_ids) > 0 and torch.cuda.is_available()
        device = torch.device(f"cuda:{int(gpu_ids[0]):d}" ) if using_cuda else torch.device('cpu')

        # torch.cuda.set_device() encouraged by pytorch developer, although dicouraged in the doc.
        # https://github.com/pytorch/pytorch/issues/70404#issuecomment-1001113109
        # It solves the bug of RNN always running on gpu 0.
        torch.cuda.set_device(device)

    else: # Multiple GPUs Distributed Training
        # print(f"trying multi-gpu: {gpu_ids}")
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            args.rank = int(os.environ["RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
            args.local_rank = int(os.environ["LOCAL_RANK"])
        elif "SLURM_PROCID" in os.environ:
            args.rank = int(os.environ["SLURM_PROCID"])
            args.local_rank = args.rank % torch.cuda.device_count()
        else:
            print("Not using distributed mode")
            args.distributed = False
            return

        args.distributed = True

        device = args.local_rank
        torch.cuda.set_device(args.local_rank)
        args.dist_backend = "nccl"
        args.dist_url = "env://"
        print(
            "| distributed init (rank {}, world {}): {}".format(
                args.rank, args.world_size, args.dist_url
            ),
            flush=True,
        )
        torch.distributed.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(
                days=365
            ),  # allow auto-downloading and de-compressing
        )
        torch.distributed.barrier()
        setup_for_distributed(args.rank == 0)

    return device #torch.device('cpu')

def count_parameters(model):
    res = 0
    for para in model.parameters():
        if para.requires_grad:
            res += para.numel() # Calculate the number of elements in the matrix
            # print(ppara) # Parameter containing
    return res

def dict_to_str(src_dict):
    dst_str = ""
    try:
        for key in src_dict.keys():
            dst_str += f" {key}: {src_dict[key]:.5f} "
        return dst_str
    except:
        for key in src_dict.keys():
            dst_str += f" {key}: {src_dict[key]} "
        return dst_str
