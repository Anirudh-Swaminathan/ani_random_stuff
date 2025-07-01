#!/usr/bin/env python3
import os
import torch

# PyTorch geometric import
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.datasets import KarateClub

"""
Introduction to PyTorch Geometric and Karate Club Dataset
Tutorial Google Colab Notebook Link: https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8
"""

# function to visualize graph
def visualize_graph(G, color, save_path=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    # position the nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    #nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=14)
    nx.draw_networkx(G,
                     pos=pos,
                     with_labels=True,
                     node_color=color,
                     #edge_color='gray',
                     node_size=700,
                     font_size=14,
                     cmap="Set2")
    plt.title("Graph Visualization")

    # save image to file if save_path is valid image file path that can be created
    if save_path and os.path.splitext(save_path)[1] in ['.png', '.jpg', '.jpeg']:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    else:
        print("No valid save path provided, displaying graph instead.")
    plt.show()

# function to visualize embeddings
def visualize_embedding(h, color, epoch=None, loss=None, save_path=None):
    plt.figure(figsize=(7, 7))
    plt.xticks([])
    plt.yticks([])

    # detach h to CPU and convert to numpy
    h = h.detach().cpu().numpy()

    # plot a scatter map of first two dimensions of h
    plt.scatter(h[:, 0],
                h[:, 1],
                c=color,
                cmap="Set2",
                s=140,
                edgecolor='k',
                alpha=0.7)

    if epoch is not None and loss is not None:
        plt.title(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    else:
        plt.title("Node Embeddings Visualization")

    plt.xlabel("Embedding Dimension 1")
    plt.ylabel("Embedding Dimension 2")

    # save image to file if save_path is valid image file path that can be created
    if save_path and os.path.splitext(save_path)[1] in ['.png', '.jpg', '.jpeg']:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
    else:
        print("No valid save path provided, displaying embedding instead.")
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

    # load the Karate Club dataset
    dataset = KarateClub()
    print(f"Dataset loaded: {dataset}")
    print(f"==========================================")
    print(f"Number of graphs in dataset: {len(dataset)}")
    print(f"Number of features per node: {dataset.num_features}")
    print(f"Number of classes: {dataset.num_classes}")
    print(f"==========================================")

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
    print(f"==========================================")

    # Gather some statistics about the graph
    print(f"Number of nodes: {data.num_nodes}")
    print(f"Number of edges: {data.num_edges}")
    print(f"Average node degree: {data.num_edges / data.num_nodes:.2f}")
    print(f"Number of training nodes: {data.train_mask.sum().item()}")
    print(f"Training node label rate: {int(data.train_mask.sum().item()) / data.num_nodes:.2f}")
    print(f"Does graph have isolated nodes?: {data.has_isolated_nodes()}")
    print(f"Does graph have self-loops?: {data.has_self_loops()}")
    print(f"Is the graph undirected?: {data.is_undirected()}")
    print(f"==========================================")

    # get the first edge object
    edge_index = data.edge_index

    # print the edge index information
    # print the first 20 edge index tensor information
    print(f"==========================================")
    print(f"First 20 elements of Edge index tensor: {edge_index.t()[:20]}")
    print(f"Edge index tensor shape: {edge_index.t().shape}")
    print(f"Edge index tensor device: {edge_index.t().device}")
    print(f"Edge index tensor dtype: {edge_index.t().dtype}")
    print(f"==========================================")


if __name__ == "__main__":
    main()

