#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# native imports
import os
from enum import Enum

# third party imports
import torch

# imports for visualization
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
import networkx as nx

# PyG imports
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
from torch_geometric.nn import GCNConv

# torch modeling imports
import torch
from torch.nn import Linear
import torch.nn.functional as F

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


def visualize_graph(G, color, save_path=None):
    """
    Function to visualize a graph using NetworkX and Matplotlib.

    @param G: A NetworkX graph object. (required)
    @param color: A list of colors for the nodes. (required)
    @param save_path: Optional path to save the visualization. If None, the plot will not be saved.
    """
    plt.figure(figsize=(10, 10))
    plt.xticks([])
    plt.yticks([])

    # position the nodes using spring layout
    pos = nx.spring_layout(G, seed=42)
    #nx.draw(graph, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=700, font_size=14)
    nx.draw_networkx(G,
                     pos=pos,
                     with_labels=False,
                     node_color=color,
                     edge_color='gray',
                     node_size=5,
                     font_size=4,
                     cmap="Set2")
    plt.title("Planetoid Cora Graph Visualization")

    # save image to file if save_path is valid image file path that can be created
    if save_path and os.path.splitext(save_path)[1] in ['.png', '.jpg', '.jpeg']:
        plt.savefig(save_path, format='png', bbox_inches='tight')
        print(f"Graph saved to {save_path}")
    else:
        print("No valid save path provided, displaying graph instead.")
    plt.show()


# define an enum for model type (MLP, GCN or GAT)
class ModelType(Enum):
    MLP = "MLP"
    GCN = "GCN"
    GAT = "GAT"

    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_

class MLP(torch.nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) model for node classification.
    """

    def __init__(self, num_features, hidden_channels, num_classes):
        # call the parent class constructor
        super().__init__()

        # set torch manual seed for reproducibility
        torch.manual_seed(12345)

        # First Linear layer
        self.lin1 = Linear(num_features, hidden_channels)

        # Second Linear layer
        self.lin2 = Linear(hidden_channels, num_classes)

    def forward(self, x):
        # Apply the first linear layer followed by ReLU activation
        x = self.lin1(x)
        x = x.relu()

        # Apply dropout for regularization
        x = F.dropout(x, p=0.5, training=self.training)

        # Apply the second linear layer
        x = self.lin2(x)
        return x

class GCN(torch.nn.Module):
    """
    A simple Graph Convolutional Network (GCN) model for node classification.
    """

    def __init__(self, num_features, hidden_channels, num_classes):
        """
        Constructor for the GCN model.

        @param num_features: The number of input features per node (required).
        @param hidden_channels: The number of hidden channels (neurons) in the GCN layer (required).
        @param num_classes: The number of output classes for classification (required).

        @return: None
        """
        # call the parent class constructor
        super().__init__()

        # set torch manual seed for reproducibility
        torch.manual_seed(1234567)

        # First graph convolutional layer
        self.conv1 = GCNConv(num_features, hidden_channels)

        # Secong graph convolutional layer
        self.conv2 = GCNConv(hidden_channels, num_classes)

    def forward(self, x, edge_index):
        """
        Forward pass of the GCN model.

        @param x: Node feature matrix of shape [num_nodes, num_features] (required).
        @param edge_index: Graph connectivity in COO format with shape [2, num_edges] (required).

        @return x: The output logits for each node of shape [num_nodes, num_classes].
        """
        # First graph convolution layer with ReLU activation and Dropout for regularization
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = F.dropout(x, p=0.5, training=self.training)

        # Second graph convolution layer
        x = self.conv2(x, edge_index)
        return x

def train(model, data, optimizer, criterion, device, mtype: ModelType = ModelType.MLP):
    """
    A function to train the model for one epoch.

    @param model: The model to train (required).
    @param data: The graph data object containing node features, edge indices, and labels (required).
    @param optimizer: The optimizer to use for updating model weights (required).
    @param criterion: The loss function for training (required).
    @param device: The device (CPU or GPU) to perform computations on (required).
    @param mtype: Model type (default is MLP).
                    This can be used to handle any model-specific logic if needed. (Optional)

    @return loss: The loss value after training for one epoch.
    """
    # set the model to training mode
    model.train()

    # set the optimizer to zero gradients
    optimizer.zero_grad()

    # perform a single forward pass of the model
    if mtype == ModelType.MLP:
        # for MLP, we only need node features
        out = model(data.x.to(device))
    elif mtype == ModelType.GCN:
        # for GCN, we need both node features and edge index
        out = model(data.x.to(device), data.edge_index.to(device))

    # compute the loss based on the training nodes only
    loss = criterion(out[data.train_mask], data.y[data.train_mask].to(device))

    # Derive gradients via backpropagation for the model parameters
    loss.backward()

    # Update the model parameters using the optimizer based on the computed gradients
    optimizer.step()
    return loss

def test(model, data, device, mtype: ModelType = ModelType.MLP):
    """
    A function to evaluate the model on the test set.

    @param model: The model to evaluate on the test set (required).
    @param data: The graph data object containing node features, edge indices, and labels (required).
    @param device: The device (CPU or GPU) to perform computations on (required).
    @param mtype: Model type (default is MLP).
                    This can be used to handle any model-specific logic if needed. (Optional)

    @return test_acc: The accuracy of the model on the test set.
    """
    # set the model to evaluation mode
    model.eval()

    # perform a forward pass without computing gradients
    with torch.no_grad():
        if mtype == ModelType.MLP:
            out = model(data.x.to(device))
        elif mtype == ModelType.GCN:
            out = model(data.x.to(device), data.edge_index.to(device))

        # get the predicted class with the highest probability by taking the argmax of the output logits
        pred = out.argmax(dim=1)

        # check against ground-truth labels for the test nodes only
        test_correct = pred[data.test_mask] == data.y[data.test_mask].to(device)

        # compute ratio of correct predictions to total test nodes
        test_acc = int(test_correct.sum()) / int(data.test_mask.sum())

    return test_acc

def train_test(data, device, num_features, num_classes, mtype: ModelType = ModelType.MLP):
    """
    A convenience function to train and test models on the Planetoid Cora dataset.
    This function encapsulates the entire training and testing process for the models.

    @param data: The graph data object containing node features, edge indices, and labels (required).
    @param device: The device (CPU or GPU) to perform computations on (required).
    @param num_features: The number of input features per node (required).
    @param num_classes: The number of output classes for classification (required).
    @param mtype: Model type (default is MLP).
                    This can be used to handle any model-specific logic if needed. (Optional)

    @return: None
    """
    # instantiate the model
    if mtype == ModelType.MLP:
        model = MLP(num_features=num_features, hidden_channels=16, num_classes=num_classes)
    elif mtype == ModelType.GCN:
        model = GCN(num_features=num_features, hidden_channels=16, num_classes=num_classes)

    model = model.to(device)

    print(f"============ {mtype.value} Model Summary ===========")
    print(f"{mtype.value} Model architecture:\n{model}")
    print(f"==========================================")

    # visualize the node embeddings before training for GCN model
    if mtype == ModelType.GCN:
        print(f"============ Visualizing {mtype.value} Model Node Embeddings Before Training ===========")
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            visualize(out, color=data.y.cpu(), save_path=f"{mtype.value}_model_node_embeddings_before_training.png")
            print(f"Saved plot of {mtype.value} model node embeddings before training to "
                  f"{mtype.value}_model_node_embeddings_before_training.png")
        print(f"==============================================")

    # set the model to training mode
    model.train()

    # Define the Loss function (CrossEntropyLoss for multi-class classification)
    criterion = torch.nn.CrossEntropyLoss()
    print(f"============ Loss Function ===========")
    print(f"Loss function: {criterion}")
    print(f"=====================================")

    # Define the optimizer (Adaptive Moment Estimation (Adam) optimizer)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    print(f"============ Optimizer ===========")
    print(f"Optimizer: {optimizer}")
    print(f"===================================")

    # Number of training epochs
    num_epochs = 150
    print(f"Number of training epochs: {num_epochs}")

    # print frequency, how often in epochs to print training progress
    print_freq = 25
    print(f"Printing training progress every {print_freq} epochs")

    # training loop
    print(f"======================= Training {mtype.value} Model =======================")
    print(f"Training {mtype.value} Model for {num_epochs} epochs...")
    for epoch in range(1, num_epochs + 1):
        # train the model for one epoch
        loss = train(model, data, optimizer, criterion, device, mtype)
        if epoch % print_freq == 0 or epoch == 1:
            print(f"Epoch: {epoch:03d}; Training Loss: {loss.item():.4f}")

    # test the model after all epochs are done
    print(f"======================= Testing {mtype.value} Model =======================")
    test_acc = test(model, data, device, mtype)
    print(f"{mtype.value} Model Test Set Accuracy: {test_acc:.4f}")

    # Visualize the node embeddings after training for GCN model
    if mtype == ModelType.GCN:
        print(f"============ Visualizing {mtype.value} Model Node Embeddings After Training ===========")
        model.eval()
        with torch.no_grad():
            out = model(data.x.to(device), data.edge_index.to(device))
            visualize(out, color=data.y.cpu(), save_path=f"{mtype.value}_model_node_embeddings_after_training.png")
            print(f"Saved plot of {mtype.value} model node embeddings after training to "
                  f"{mtype.value}_model_node_embeddings_after_training.png")
        print(f"==============================================")



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
    # store the dataset in the data/Planetoid directory
    # if the dataset is not already present in the directory, it will be downloaded automatically
    # NormalizeFeatures transform is used to min-shift row-normalize the
    # bag-of-words input node feature vectors to sum to one
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

    # convert the graph to networkx
    G = to_networkx(data, to_undirected=True, remove_self_loops=True)

    # visualize the graph and save it to a file
    visualize_graph(G, color=data.y.cpu().numpy(), save_path="planetoid_cora_graph.png")

    # call the train test loop for MLP model
    train_test(data, device, num_features=dataset.num_features, num_classes=dataset.num_classes, mtype=ModelType.MLP)

    # call the train test loop for GCN model
    train_test(data, device, num_features=dataset.num_features, num_classes=dataset.num_classes, mtype=ModelType.GCN)


if __name__ == "__main__":
    main()
