import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing


class MyGNN(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MyGNN, self).__init__(aggr='mean')
        self.lin1 = nn.Linear(in_channels, out_channels)
        self.lin2 = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        x1, x2 = x  # Separate node feature tensors for each node type
        x1 = self.lin1(x1)
        x2 = self.lin2(x2)
        x = [x1, x2]  # Combine updated node features
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # Compute messages based on node features
        return x_j

    def update(self, aggr_out):
        # Update node features based on aggregated messages
        return aggr_out


# Example usage
node_type1_feats = torch.tensor([[1., 2.], [3., 4.]])  # Node type 1 features
node_type2_feats = torch.tensor([[5., 6.], [7., 8.]])  # Node type 2 features
edge_index = torch.tensor([[0, 1], [1, 0]])  # Example edge indices

gnn = MyGNN(in_channels=2, out_channels=2)

# Forward pass through the GNN
output = gnn([node_type1_feats, node_type2_feats], edge_index)

# Perform downstream tasks with the output
