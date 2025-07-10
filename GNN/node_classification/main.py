#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# native imports
import os

# third party imports
import torch

# imports for visualization
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

# PyG imports
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

"""
Node Classification with Graph Neural Networks (GNNs)
Tutorial Google Colab Notebook Link: https://colab.research.google.com/drive/14OvFnAXggxB8vM4e8vSURUp1TaKnovzX
"""

def visualize(h, color, save_path=None):
    """
    Visualizes the node embeddings using t-SNE.

    @param h: Node embeddings (torch.Tensor).
    @param color: Color labels for the nodes (torch.Tensor).
    @param save_path: Optional path to save the visualization. If None, the plot will not be saved.

    @return: None
    """
    # perform t-SNE on the node embeddings
    # detach() first before cpu() .
    # Moving to CPU before detaching is less optimal due to the overhead of autograd tracking the device transfer which is not something we want.
    # So, detach() before cpu()
    # numpy() requires the tensor to be on CPU, so cpu() before numpy()
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    # create a matplotlib figure and scatter plot the t-SNE results
    plt.figure(figsize=(10, 10))

    # xticks and yticks are set to empty lists to remove them
    plt.xticks([])
    plt.yticks([])

    # set xlabel and ylabel
    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")

    # set title for the plot
    plt.title("Node Embeddings Visualization with t-SNE")

    # scatter plot the t-SNE results with colors based on the labels
    plt.scatter(z[:, 0], z[:, 1], c=color, s=70, cmap='Set2', alpha=0.95)

    # save image to file if save_path is valid image file path that can be created
    if save_path and os.path.splitext(save_path)[1] in ['.png', '.jpg', '.jpeg']:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Node embeddings visualization saved to {save_path}")
    else:
        print("No valid save path provided, displaying node embeddings instead.")
    plt.show()


def main():
    # capture torch version
    torch_version = torch.__version__

    # set OS environment variable with torch version
    os.environ['TORCH'] = torch_version

    # print the torch version
    print(f"Using PyTorch version: {torch_version}")

    # multi-GPU support
    if torch.cuda.is_available():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available, using CPU.")

    # load the Planetoid dataset (Cora)
    dataset = Planetoid(root="data/Planetoid", name="Cora", transform=NormalizeFeatures())

    # print dataset information
    print(f"Dataset loaded: {dataset}")
    print(f"============ Dataset Information ============")
    print(f"Number of graphs in the dataset: {len(dataset)}")
    print(f"Number of features per node: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"=============================================")

    # Get the first graph object.
    data = dataset[0]

    # transfer the data to the device (GPU or CPU)
    data = data.to(device)

    # print the graph information
    # The data object holds 4 attributes:
    # - x: node feature vectors; a tensor of shape [num_nodes, num_features]
    # - edge_index: holds information about the graph connectivity; a tuple of source and
    #               destination node indices for each edge
    # - y: node labels; a tensor of shape [num_nodes] with class indices where each node
    #      is assigned to exactly one class
    # - train_mask: mask for training nodes; a boolean tensor of shape [num_nodes] where True indicates
    #              that the node is part of the training set
    print(f"Graph data: {data}")
    print(f"Type of graph data: {type(data)}")
    print('===========================================================================================================')

    # Gather some statistics about the graph
    print(f"Number of nodes in the graph: {data.num_nodes}")
    print(f"Number of edges in the graph: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Number of classes: {data.y.max().item() + 1}")
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
    print(f"Training node label rate: {int(data.train_mask.sum().item()) / data.num_nodes:.2%}")
    print(f"Number of validation nodes: {data.val_mask.sum().item()}")
    print(f"Validation node label rate: {int(data.val_mask.sum().item()) / data.num_nodes:.2%}")
    print(f"Number of test nodes: {data.test_mask.sum().item()}")
    print(f"Test node label rate: {int(data.test_mask.sum().item()) / data.num_nodes:.2%}")
    print(f"Does graph have isolated nodes?: {data.has_isolated_nodes()}")
    print(f"Does graph have self-loops?: {data.has_self_loops()}")
    print(f"Is the graph undirected?: {data.is_undirected()}")
    print('===========================================================================================================')

    # get the first edge object
    edge_index = data.edge_index

    # print the edge index information
    # print the first 20 edge index tensor information
    print(f"=============== Edge Index Information ===============")
    print(f"First 20 elements of Edge index tensor: {edge_index.t()[:20]}")
    print(f"Edge index tensor shape: {edge_index.t().shape}")
    print(f"Edge index tensor device: {edge_index.t().device}")
    print(f"Edge index tensor dtype: {edge_index.t().dtype}")
    print(f"======================================================")


if __name__ == "__main__":
    main()

