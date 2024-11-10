import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool
import torch.linalg as LA

class PoseGCN(torch.nn.Module):
    def __init__(self, num_features: int, hidden_dim1: int, hidden_dim2: int, output_dim: int) -> None:
        super(PoseGCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_dim1)
        self.conv2 = GCNConv(hidden_dim1, hidden_dim2)
        self.conv3 = GCNConv(hidden_dim2, output_dim)

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
        x, edge_index, batch = (
            data.x.to(self.device),
            data.edge_index.to(self.device),
            data.batch.to(self.device),
        )

        adj_matrix = torch.zeros((data.num_nodes, data.num_nodes), device=self.device)
        adj_matrix[edge_index[0], edge_index[1]] = 1

        laplacian = self.compute_laplacian(adj_matrix)
        if self.fourier_basis is None or self.fourier_basis.shape[0] != data.num_nodes:
            self.fourier_basis = self.compute_gft_basis(laplacian)

        # Apply Graph Fourier Transform (GFT) to node features
        x = self.gft(x, self.fourier_basis)

        # GCN Layers in Spectral Domain
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = torch.relu(x)
        x = self.conv3(x, edge_index)

        # Apply Inverse GFT (iGFT) to return to spatial domain
        x = self.inverse_gft(x, self.fourier_basis)

        x = global_mean_pool(x, batch)
        return x
