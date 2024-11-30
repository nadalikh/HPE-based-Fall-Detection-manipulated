import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool


class PoseGAT(torch.nn.Module):
    def __init__(
        self, num_features: int, hidden_dim1: int, hidden_dim2: int, output_dim: int, heads: int = 1
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of features in the input sequence
        hidden_dim1 : int
            Dimension of the first hidden layer of the GAT
        hidden_dim2 : int
            Dimension of the second hidden layer of the GAT
        output_dim : int
            Dimension of the output layer of the GAT
        heads : int
            Number of attention heads for the GAT layers (default: 1)
        """
        super(PoseGAT, self).__init__()
        self.conv1 = GATConv(num_features, hidden_dim1, heads=heads)
        self.conv2 = GATConv(hidden_dim1 * heads, hidden_dim2, heads=heads)
        self.conv3 = GATConv(hidden_dim2 * heads, output_dim, heads=heads, concat=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Parameters
        ----------
        data : Data
            Pose Graph

        Returns
        -------
        torch.Tensor
            Output of the GAT of shape (batch_size, output_dim)
        """
        x, edge_index, batch = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.batch.to(self.device),
        )

        # First GAT layer with attention heads
        x = self.conv1(x, edge_index)
        x = torch.relu(x)

        # Second GAT layer
        x = self.conv2(x, edge_index)
        x = torch.relu(x)

        # Third GAT layer (output layer, no ReLU)
        x = self.conv3(x, edge_index)

        # Global mean pooling for graph-level representation
        x = global_mean_pool(x, batch)
        return x
