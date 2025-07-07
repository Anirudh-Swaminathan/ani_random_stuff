#!/usr/bin/env python3
import os
import torch

# PyTorch geometric import
import networkx as nx
from matplotlib import pyplot as plt
from torch_geometric.datasets import KarateClub
from torch_geometric.utils import to_networkx

# Model architecture related imports
from torch.nn import Linear
from torch_geometric.nn import GCNConv

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
def visualize_embedding(fig, h, color, subidx: tuple=(1, 1, 1), epoch=None, loss=None):
    """
    Visualizes the node embeddings in a 2D scatter plot.

    @param fig: The figure object to plot on (required)
    @oaram h: The node embeddings tensor (required)
    @param color: The color array for the nodes (required)
    @param subidx: tuple; Subplot index for multi-plotting (default (1, 1, 1))
    @param epoch: The current training epoch (optional)
    @param loss: The current loss value (optional)

    @return: None

    @raises ValueError: If subidx is not a valid subplot index.
    """
    # Check if the subidx is a valid subplot index
    if not isinstance(subidx, (tuple, str)):
        raise ValueError("subidx must be a tuple or a string representing subplot index.")
    if isinstance(subidx, tuple):
        assert len(subidx) == 3, "subidx must be a tuple of length 3 (nrows, ncols, index)."

    # Add a subplot at subplot index subidx to the figure fig
    ax = fig.add_subplot(*subidx)

    # detach h to CPU and convert to numpy
    h = h.detach().cpu().numpy()

    # plot a scatter map of first two dimensions of h
    ax.scatter(h[:, 0],
                h[:, 1],
                c=color,
                cmap="Set2",
                s=140,
                edgecolor='k',
                alpha=0.7)

    if epoch is not None and loss is not None:
        ax.set_title(f"Epoch: {epoch}, Loss: {loss.item():.4f}", fontsize=16)
    else:
        ax.set_title("Node Embeddings Visualization")

    ax.set_xlabel("Embedding Dimension 1")
    ax.set_ylabel("Embedding Dimension 2")


class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network (GCN) model for node classification.
    """
    def __init__(self, num_features, num_classes):
        # call the parent constructor
        super().__init__()

        # set torch manual seed for reproducibility
        torch.manual_seed(1234)

        # First graph convolution layer
        self.conv1 = GCNConv(num_features, 4)

        # Second graph convolution layer
        self.conv2 = GCNConv(4, 4)

        # third graph convolution layer
        self.conv3 = GCNConv(4, 2)

        # Linear layer for final classification output
        self.classifier = Linear(2, num_classes)

    def forward(self, x, edge_index):
        # First graph convolution layer with tanh activation
        h = self.conv1(x, edge_index)
        h = h.tanh()

        # Second graph convolution layer with tanh activation
        h = self.conv2(h, edge_index)
        h = h.tanh()

        # Third graph convolution layer with tanh activation
        # This yields the final GNN embedding space
        h = self.conv3(h, edge_index)
        h = h.tanh()

        # Apply a final (linear) classifier to the GNN embedding space
        out = self.classifier(h)

        # Return the output and the intermediate node embeddings
        return out, h

def train(model, data, optimizer, criterion, device):
    """
    A function to train the GCN model on the Karate Club dataset.

    @param model: The GCN model to train (required)
    @param data: The graph data object containing node features and edge indices (required)
    @param optimizer: The optimizer for updating model weights (required)
    @param criterion: The loss function for training (required)
    @param device: The device (CPU or GPU) to perform computations on (required)

    @return: The loss value and embeddings after training for one epoch.
    """
    # set the optimizer to zero gradients
    optimizer.zero_grad()

    # perform a single forward pass of the model
    out, h = model(data.x.to(device), data.edge_index.to(device))

    # compute the loss solely based on the training nodes
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))

    # Derive gradients for the model parameters
    loss.backward()

    # Update the model parameters using the optimizer based on the computed gradients
    optimizer.step()
    return loss, h

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

    # convert the graph to networkx
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)

    # visualize the graph and save it to a file
    visualize_graph(G, color=data.y.cpu().numpy(), save_path="karate_club_graph.png")

    # instantiate the GCN model
    model = GCN(num_features=dataset.num_features, num_classes=dataset.num_classes).to(device)
    print(f"===========================================")
    print(f"Model architecture:\n{model}")
    print(f"===========================================")

    # Compute initial embedding, visualize it, and save to file
    # Even without training, with completely random initial weights,
    # nodes of the same class would be clustered together in the embedding space
    # This means GNNs introduce a strong inductive bias, leading to similar embeddings
    # for nodes that are close to each other in the input graph.
    with torch.no_grad():
        _, h = model(data.x, edge_index)
        print(f"============================================")
        print(f"Initial node embeddings shape: {h.shape}")
        print(f"Initial node embeddings device: {h.device}")
        print(f"Initial node embeddings dtype: {h.dtype}")
        print(f"============================================")

        # create a figure for visualization
        fig = plt.figure(figsize=(7, 7))
        visualize_embedding(fig, h, color=data.y.cpu().numpy())
        save_path = "karate_club_initial_embeddings.png"
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Embedding visualization saved to {save_path}")
        plt.show()

    # Set the model to training mode
    model.train()

    # Define the Loss function (CrossEntropyLoss for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"============================================")
    print(f"Loss function: {criterion}")

    # Define the optimizer (Adaptive Moment Estimation (Adam) optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    print(f"Optimizer: {optimizer}")

    # Number of training epochs
    num_epochs = 400
    print(f"Number of training epochs: {num_epochs}")

    # Visualization frequency for node embeddings
    vis_freq = 50
    print(f"Visualizing node embeddings every {vis_freq} epochs")

    # number of visualizations counter
    n_vis = 0

    # create a figure for visualization
    fig = plt.figure(figsize=(7, 7))

    print(f"============================================")
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # train the model for one epoch
        loss, h = train(model, data, optimizer, criterion, device)

        # if epoch is a multiple of vis_freq, visualize the embeddings
        if epoch % vis_freq == 0:
            # subplot index is based on the number of visualizations previously done
            n_vis += 1
            subidx = (n_vis, 1, n_vis)

            print(f"Epoch: {epoch}, Loss: {loss.item():.4f}")
            print(f"Plotting node embeddings for epoch {epoch}...")

            # visualize the embeddings
            visualize_embedding(fig, h, color=data.y.cpu().numpy(), subidx=subidx, epoch=epoch, loss=loss)
    print(f"Training completed after {num_epochs} epochs.")
    print(f"============================================")

    # save the embedding visualization to a file
    save_path = "karate_club_training_progress.png"
    plt.savefig(save_path, format='png', bbox_inches='tight')
    print(f"Embedding visualization for training progress saved to {save_path}")
    plt.show()
    print(f"============================================")


if __name__ == "__main__":
    main()

