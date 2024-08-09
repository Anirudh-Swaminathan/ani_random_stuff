#!/usr/bin/env python3
"""
Program to utilize 2 GPUs, if available with PyTorch DistributedDataParallel
"""
import os

# native imports here
# 3rd party imports here
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP


def example(rank, world_size):
    print(f"Passed args are\nrank: {rank}\nworld_size: {world_size}")

    # create default process group
    print(f"Initializing Distributed process group!")
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

    # device
    cuda_counts = torch.cuda.device_count()
    print(f"Torch CUDA device count is {cuda_counts}")
    device = rank
    if cuda_counts > 1:
        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    # create local model
    model = nn.Linear(10, 10).to(device)
    print(f"Model is\n{model}")

    # construct DDP model
    ddp_model = DDP(model, device_ids=[rank])
    print(f"DDP model is\n{ddp_model}")

    # define loss function
    loss_fn = nn.MSELoss()
    print(f"Loss function is\n{loss_fn}")

    # define optimizer
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)
    print(f"optimizer is\n{optimizer}")

    # forward pass
    outputs = ddp_model(torch.randn(20, 10).to(device))
    print(f"ddp_model outputs is of size {outputs.size()}")
    print(f"ddp_model outputs is\n{outputs}")

    # create dummy labels
    labels = torch.randn(20, 10).to(device)
    print(f"labels is of size {labels.size()}")
    print(f"labels is\n{labels}")

    # backward pass
    loss_fn(outputs, labels).backward()

    # update parameters
    optimizer.step()


def main():
    # initialize world_size
    world_size = 2
    print(f"World size is {world_size}")

    # spawn multiprocessor threads/processes
    print(f"About to run the statement mp.spawn")
    mp.spawn(
        example,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )
    print(f"Just ran the statement mp.spawn")


if __name__ == "__main__":
    # Environment variables which need to be
    # set when using c10d's default "env"
    # initialization mode.
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    main()
