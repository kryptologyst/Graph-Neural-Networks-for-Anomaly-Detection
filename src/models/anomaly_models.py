"""Anomaly detection models for graph neural networks."""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GAE, VGAE, GATConv, SAGEConv
from torch_geometric.nn.models import InnerProductDecoder


class GCNEncoder(nn.Module):
    """Graph Convolutional Network encoder for anomaly detection.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output features.
        num_layers: Number of GCN layers.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GATEncoder(nn.Module):
    """Graph Attention Network encoder for anomaly detection.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output features.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GATConv(in_channels, hidden_channels, heads=num_heads, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * num_heads, out_channels, heads=1, dropout=dropout)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x


class GraphSAGEEncoder(nn.Module):
    """GraphSAGE encoder for anomaly detection.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output features.
        num_layers: Number of SAGE layers.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class VariationalGCNEncoder(nn.Module):
    """Variational GCN encoder for VGAE.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output features.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv_mu = GCNConv(hidden_channels, out_channels)
        self.conv_logvar = GCNConv(hidden_channels, out_channels)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Tuple of (mean, log variance) for variational inference.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.conv_mu(x, edge_index), self.conv_logvar(x, edge_index)


class DOMINANT(nn.Module):
    """DOMINANT: Deep Anomaly Detection on Attributed Networks.
    
    A more sophisticated anomaly detection model that combines reconstruction
    loss for both structure and attributes.
    
    Args:
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output features.
        dropout: Dropout rate.
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dropout = dropout
        
        # Encoder
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        
        # Decoder for structure
        self.struct_decoder = InnerProductDecoder()
        
        # Decoder for attributes
        self.attr_decoder = nn.Linear(out_channels, in_channels)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Encode node features.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Node embeddings.
        """
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x
    
    def decode_struct(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Decode structure from embeddings.
        
        Args:
            z: Node embeddings.
            edge_index: Edge indices.
            
        Returns:
            Reconstructed adjacency matrix.
        """
        return self.struct_decoder(z, edge_index)
    
    def decode_attr(self, z: torch.Tensor) -> torch.Tensor:
        """Decode attributes from embeddings.
        
        Args:
            z: Node embeddings.
            
        Returns:
            Reconstructed node features.
        """
        return self.attr_decoder(z)
    
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Tuple of (embeddings, reconstructed structure, reconstructed attributes).
        """
        z = self.encode(x, edge_index)
        struct_recon = self.decode_struct(z, edge_index)
        attr_recon = self.decode_attr(z)
        return z, struct_recon, attr_recon
    
    def compute_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        alpha: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute DOMINANT loss.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            pos_edge_index: Positive edge indices.
            neg_edge_index: Negative edge indices.
            alpha: Weight for structure vs attribute reconstruction.
            
        Returns:
            Tuple of (total loss, structure loss, attribute loss).
        """
        z, struct_recon, attr_recon = self.forward(x, edge_index)
        
        # Structure reconstruction loss
        pos_loss = -torch.log(struct_recon + 1e-15).mean()
        neg_loss = -torch.log(1 - struct_recon + 1e-15).mean()
        struct_loss = pos_loss + neg_loss
        
        # Attribute reconstruction loss
        attr_loss = F.mse_loss(attr_recon, x)
        
        # Total loss
        total_loss = alpha * struct_loss + (1 - alpha) * attr_loss
        
        return total_loss, struct_loss, attr_loss
    
    def anomaly_score(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Compute anomaly scores for nodes.
        
        Args:
            x: Node features.
            edge_index: Edge indices.
            
        Returns:
            Anomaly scores for each node.
        """
        self.eval()
        with torch.no_grad():
            z, struct_recon, attr_recon = self.forward(x, edge_index)
            
            # Combine structure and attribute reconstruction errors
            struct_error = torch.norm(struct_recon - torch.eye(x.size(0), device=x.device), dim=1)
            attr_error = torch.norm(attr_recon - x, dim=1)
            
            # Normalize and combine
            struct_error = (struct_error - struct_error.mean()) / struct_error.std()
            attr_error = (attr_error - attr_error.mean()) / attr_error.std()
            
            anomaly_scores = 0.5 * struct_error + 0.5 * attr_error
            
        return anomaly_scores


def create_model(
    model_name: str,
    in_channels: int,
    hidden_channels: int = 64,
    out_channels: int = 32,
    **kwargs
) -> nn.Module:
    """Create an anomaly detection model.
    
    Args:
        model_name: Name of the model to create.
        in_channels: Number of input features.
        hidden_channels: Number of hidden units.
        out_channels: Number of output features.
        **kwargs: Additional model parameters.
        
    Returns:
        PyTorch model.
    """
    if model_name.lower() == "gae":
        encoder = GCNEncoder(in_channels, hidden_channels, out_channels, **kwargs)
        return GAE(encoder)
    
    elif model_name.lower() == "vgae":
        encoder = VariationalGCNEncoder(in_channels, hidden_channels, out_channels, **kwargs)
        return VGAE(encoder)
    
    elif model_name.lower() == "gat_gae":
        encoder = GATEncoder(in_channels, hidden_channels, out_channels, **kwargs)
        return GAE(encoder)
    
    elif model_name.lower() == "sage_gae":
        encoder = GraphSAGEEncoder(in_channels, hidden_channels, out_channels, **kwargs)
        return GAE(encoder)
    
    elif model_name.lower() == "dominant":
        return DOMINANT(in_channels, hidden_channels, out_channels, **kwargs)
    
    else:
        raise ValueError(f"Unknown model: {model_name}")
