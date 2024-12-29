import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool
from torch_geometric.data import Batch
import torch.linalg as LA

class HybridGATGCNWithIGFT(torch.nn.Module):
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
        super(HybridGATGCNWithIGFT, self).__init__()

        # Define GCN layer
        self.gcn_conv = GCNConv(num_features, gcn_hidden_dim)

        # Define GAT layer
        self.gat_conv = GATConv(gcn_hidden_dim, gat_hidden_dim, heads=gat_heads)

        # Define output layer
        self.output_layer = torch.nn.Linear(gat_hidden_dim * gat_heads, output_dim)

        # Fourier basis (eigenvectors) will be stored here
        self.fourier_basis = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def compute_laplacian(self, adj_matrix: torch.Tensor) -> torch.Tensor:
        degree_matrix = torch.diag(torch.sum(adj_matrix, dim=1))
        laplacian = degree_matrix - adj_matrix
        return laplacian

    def compute_gft_basis(self, laplacian: torch.Tensor) -> torch.Tensor:
        eigenvalues, eigenvectors = LA.eigh(laplacian)  # Compute eigenvalues and eigenvectors
        return eigenvectors

    def gft(self, x: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        if eigenvectors.shape[0] != x.shape[0]:
            raise ValueError(f"GFT Error: Eigenvector count {eigenvectors.shape[0]} does not match node count {x.shape[0]}")
        return torch.matmul(eigenvectors.T, x)

    def inverse_gft(self, x: torch.Tensor, eigenvectors: torch.Tensor) -> torch.Tensor:
        if eigenvectors.shape[0] != x.shape[0]:
            raise ValueError(f"iGFT Error: Eigenvector count {eigenvectors.shape[0]} does not match node count {x.shape[0]}")
        return torch.matmul(eigenvectors, x)

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

        # Compute adjacency matrix
        adj_matrix = torch.zeros((data.num_nodes, data.num_nodes), device=self.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1

        # Compute Laplacian and Fourier basis if not already computed
        laplacian = self.compute_laplacian(adj_matrix)
        if self.fourier_basis is None or self.fourier_basis.shape[0] != data.num_nodes:
            self.fourier_basis = self.compute_gft_basis(laplacian)

        # Apply GFT to node features
        x = self.gft(x, self.fourier_basis)

        # GCN forward pass
        x = self.gcn_conv(x, edge_index)
        x = F.relu(x)

        # GAT forward pass
        x = self.gat_conv(x, edge_index)
        x = F.relu(x)

        # Apply iGFT to return to spatial domain
        x = self.inverse_gft(x, self.fourier_basis)

        # Output layer
        x = global_mean_pool(x, batch)
        x = self.output_layer(x)
        return x
