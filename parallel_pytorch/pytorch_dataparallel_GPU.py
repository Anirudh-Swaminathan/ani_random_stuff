#!/usr/bin/env python3
"""
Program to utilize 2 GPUs, if available with PyTorch Parallel
"""
# native imports here
# 3rd party imports here
import torch
import torch.nn as nn
from torch.nn import DataParallel as DP
from torch.utils.data import Dataset, DataLoader


class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, size)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.len


class Model(nn.Module):
    # Custom model

    def __init__(self, input_size, output_size):
        super(Model, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, input):
        output = self.fc(input)
        print("\tIn Model: input size", input.size(), "output size", output.size())

        return output


def main():
    # Parameters and DataLoaders
    input_size = 5
    output_size = 2

    batch_size = 30
    data_size = 100

    print(f"Defined parameters\ninput_size: {input_size}\noutput_size: {output_size}\nbatch_size: {batch_size}\ndata_size: {data_size}")

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device is {device}")

    # data loader
    rand_loader = DataLoader(dataset=RandomDataset(input_size, data_size), batch_size=batch_size, shuffle=True)
    print(f"Data loader is {rand_loader}")

    # create model
    model = Model(input_size, output_size)
    print(f"Model is {model}")
    print(f"Torch CUDA device count is {torch.cuda.device_count()}")

    # if multiple CUDA devices exist
    if torch.cuda.device_count() > 1:
        print(f"Let's use all {torch.cuda.device_count()} GPUs!")
        # dim = 0 [30, ...] -> [10, ...], [10, ...] -> [10, ...] on 3 GPUs for example
        model = DP(model)
        print(f"After wrapping with DataParallel, model is {model}")
    else: print(f"Model was not wrapped with DataParallel because of lack of multiple GPUs!")

    # move model to CUDA
    model = model.to(device)
    print(f"After moving model to {device}, it is now {model}")

    # Run the model
    for data in rand_loader:
        b_input = data.to(device)
        b_output = model(b_input)
        print(f"Outside: input size {b_input.size()}, output_size {b_output.size()}")


if __name__ == "__main__":
    main()

