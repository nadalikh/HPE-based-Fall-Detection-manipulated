import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Batch


class HybridGATGCN(torch.nn.Module):
    def __init__(
        self,
        num_features: int,
        gcn_hidden_dim: int,
        gat_hidden_dim: int,
        output_dim: int,
        gat_heads: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        num_features : int
            Number of input features.
        gcn_hidden_dim : int
            Dimension of the hidden GCN layer.
        gat_hidden_dim : int
            Dimension of the hidden GAT layer.
        output_dim : int
            Dimension of the output layer.
        gat_heads : int
            Number of attention heads for GAT (default: 1).
        """
        super(HybridGATGCN, self).__init__()
        
        # Define GCN layer
        self.gcn_conv = GCNConv(num_features, gcn_hidden_dim)

        # Define GAT layer
        self.gat_conv = GATConv(gcn_hidden_dim, gat_hidden_dim, heads=gat_heads)

        # Define output layer
        self.output_layer = torch.nn.Linear(gat_hidden_dim * gat_heads, output_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data: Batch) -> torch.Tensor:
        """
        Parameters
        ----------
        data : Batch
            Batched graph data.

        Returns
        -------
        torch.Tensor
            Output of the hybrid model.
        """
        x, edge_index, batch = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.batch.to(self.device),
        )
        edge_attr = (
            data.edge_attr.to(self.device)
            if hasattr(data, "edge_attr") and data.edge_attr is not None
            else None
        )

        # GCN forward pass
        x = self.gcn_conv(x, edge_index)
        x = F.relu(x)

        # GAT forward pass
        x = self.gat_conv(x, edge_index)
        x = F.relu(x)

        # Output layer
        x = global_mean_pool(x, batch)
        x = self.output_layer(x)
        return x
