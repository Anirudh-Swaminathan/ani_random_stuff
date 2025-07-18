# Introduction: Hands-on Graph Neural Networks

This folder contains the code for the first tutorial in the PyTorch Geometric series, which provides a hands-on introduction to Graph Neural Networks (GNNs).

The link to this Google Colab Notebook for this tutorial is [Introduction: Hands-on Graph Neural Networks](https://colab.research.google.com/drive/1h3-vJGRVloF5zStxL5I0rSy4ZUPNsjy8)

## Tutorial Outputs

The python code output is as follows:

```bash
python main.py
```

```plaintext
Using PyTorch version: 2.0.1+cu117
Using device: cuda
Dataset loaded: KarateClub()
==========================================
Number of graphs in dataset: 1
Number of features per node: 34
Number of classes: 4
==========================================
Graph data: Data(x=[34, 34], edge_index=[2, 156], y=[34], train_mask=[34])
Type of graph data: <class 'torch_geometric.data.data.Data'>
==========================================
Number of nodes: 34
Number of edges: 156
Average node degree: 4.59
Number of training nodes: 4
Training node label rate: 0.12
Does graph have isolated nodes?: False
Does graph have self-loops?: False
Is the graph undirected?: True
==========================================
==========================================
First 20 elements of Edge index tensor: tensor([[ 0,  1],
        [ 0,  2],
        [ 0,  3],
        [ 0,  4],
        [ 0,  5],
        [ 0,  6],
        [ 0,  7],
        [ 0,  8],
        [ 0, 10],
        [ 0, 11],
        [ 0, 12],
        [ 0, 13],
        [ 0, 17],
        [ 0, 19],
        [ 0, 21],
        [ 0, 31],
        [ 1,  0],
        [ 1,  2],
        [ 1,  3],
        [ 1,  7]], device='cuda:0')
Edge index tensor shape: torch.Size([156, 2])
Edge index tensor device: cuda:0
Edge index tensor dtype: torch.int64
==========================================
Graph saved to karate_club_graph.png
===========================================
Model architecture:
GCN(
  (conv1): GCNConv(34, 4)
  (conv2): GCNConv(4, 4)
  (conv3): GCNConv(4, 2)
  (classifier): Linear(in_features=2, out_features=4, bias=True)
)
===========================================
============================================
Initial node embeddings shape: torch.Size([34, 2])
Initial node embeddings device: cuda:0
Initial node embeddings dtype: torch.float32
============================================
Embedding visualization saved to karate_club_initial_embeddings.png
============================================
Loss function: CrossEntropyLoss()
Optimizer: Adam (
Parameter Group 0
    amsgrad: False
    betas: (0.9, 0.999)
    capturable: False
    differentiable: False
    eps: 1e-08
    foreach: None
    fused: None
    lr: 0.01
    maximize: False
    weight_decay: 0
)
Number of training epochs: 400
Visualizing node embeddings every 50 epochs
Total number of visualizations: 8
============================================
Starting training for 400 epochs...
Epoch: 50, Loss: 0.5809
Plotting node embeddings for epoch 50 at subplot index (8, 1, 1)
Epoch: 100, Loss: 0.1823
Plotting node embeddings for epoch 100 at subplot index (8, 1, 2)
Epoch: 150, Loss: 0.1009
Plotting node embeddings for epoch 150 at subplot index (8, 1, 3)
Epoch: 200, Loss: 0.0671
Plotting node embeddings for epoch 200 at subplot index (8, 1, 4)
Epoch: 250, Loss: 0.0489
Plotting node embeddings for epoch 250 at subplot index (8, 1, 5)
Epoch: 300, Loss: 0.0376
Plotting node embeddings for epoch 300 at subplot index (8, 1, 6)
Epoch: 350, Loss: 0.0300
Plotting node embeddings for epoch 350 at subplot index (8, 1, 7)
Epoch: 400, Loss: 0.0247
Plotting node embeddings for epoch 400 at subplot index (8, 1, 8)
Training completed after 400 epochs.
============================================
Embedding visualization for training progress saved to karate_club_training_progress.png
============================================
```

A visualization of the graph is found in this image output:

![Karate Club Graph](./karate_club_graph.png)

### Initial Node Embeddings

Even without training with completely random initial weights, nodes of the same class would be clustered together in the embedding space.
This means GNNs introduce a strong inductive bias, leading to similar embeddings for nodes that are close to each other in the input graph.

A visualization of the initial embedding of the graph is found in this image output:

![Karate Club Initial Embeddings](./karate_club_initial_embeddings.png)

### Evolution of Node Embeddings through Training

After training the GCN for 400 epochs, the node embeddings evolve to better represent the classes of the nodes in the graph.

A visualization of the evolution of the node embeddings during training over multiple epochs is found in this image output:

![Karate Club Node Embeddings Evolution](./karate_club_training_progress.png)

