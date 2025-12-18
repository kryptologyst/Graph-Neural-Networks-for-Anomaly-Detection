"""Unit tests for anomaly detection models."""

import pytest
import torch
import numpy as np

from src.models.anomaly_models import (
    GCNEncoder,
    GATEncoder,
    GraphSAGEEncoder,
    VariationalGCNEncoder,
    DOMINANT,
    create_model,
)
from src.utils.device import get_device, set_seed
from src.data.dataset import SyntheticAnomalyDataset, load_dataset
from src.eval.metrics import AnomalyEvaluator, compute_anomaly_scores


class TestModels:
    """Test model implementations."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.in_channels = 16
        self.hidden_channels = 32
        self.out_channels = 16
        self.num_nodes = 100
        self.num_edges = 200
        
        # Create dummy data
        self.x = torch.randn(self.num_nodes, self.in_channels)
        self.edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges))
        self.pos_edge_index = self.edge_index[:, :self.num_edges//2]
        self.neg_edge_index = torch.randint(0, self.num_nodes, (2, self.num_edges//2))
    
    def test_gcn_encoder(self):
        """Test GCN encoder."""
        encoder = GCNEncoder(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
        )
        
        output = encoder(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        assert not torch.isnan(output).any()
    
    def test_gat_encoder(self):
        """Test GAT encoder."""
        encoder = GATEncoder(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
        )
        
        output = encoder(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        assert not torch.isnan(output).any()
    
    def test_sage_encoder(self):
        """Test GraphSAGE encoder."""
        encoder = GraphSAGEEncoder(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
        )
        
        output = encoder(self.x, self.edge_index)
        assert output.shape == (self.num_nodes, self.out_channels)
        assert not torch.isnan(output).any()
    
    def test_variational_encoder(self):
        """Test variational GCN encoder."""
        encoder = VariationalGCNEncoder(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
        )
        
        mu, logvar = encoder(self.x, self.edge_index)
        assert mu.shape == (self.num_nodes, self.out_channels)
        assert logvar.shape == (self.num_nodes, self.out_channels)
        assert not torch.isnan(mu).any()
        assert not torch.isnan(logvar).any()
    
    def test_dominant(self):
        """Test DOMINANT model."""
        model = DOMINANT(
            self.in_channels,
            self.hidden_channels,
            self.out_channels,
        )
        
        z, struct_recon, attr_recon = model(self.x, self.edge_index)
        assert z.shape == (self.num_nodes, self.out_channels)
        assert struct_recon.shape == (self.num_nodes, self.num_nodes)
        assert attr_recon.shape == (self.num_nodes, self.in_channels)
        
        # Test loss computation
        loss, struct_loss, attr_loss = model.compute_loss(
            self.x, self.edge_index, self.pos_edge_index, self.neg_edge_index
        )
        assert loss.item() > 0
        assert struct_loss.item() > 0
        assert attr_loss.item() > 0
        
        # Test anomaly scoring
        scores = model.anomaly_score(self.x, self.edge_index)
        assert scores.shape == (self.num_nodes,)
        assert not torch.isnan(scores).any()
    
    def test_create_model(self):
        """Test model creation function."""
        models = ["gae", "vgae", "gat_gae", "sage_gae", "dominant"]
        
        for model_name in models:
            model = create_model(
                model_name=model_name,
                in_channels=self.in_channels,
                hidden_channels=self.hidden_channels,
                out_channels=self.out_channels,
            )
            assert model is not None


class TestDataset:
    """Test dataset functionality."""
    
    def test_synthetic_dataset(self):
        """Test synthetic dataset creation."""
        dataset = SyntheticAnomalyDataset(
            root="./test_data",
            num_nodes=100,
            graph_type="barabasi_albert",
            anomaly_ratio=0.1,
            anomaly_type="structural",
        )
        
        data = dataset[0]
        assert data.num_nodes == 100
        assert data.num_edges > 0
        assert hasattr(data, 'anomaly_labels')
        assert data.anomaly_labels.sum().item() == 10  # 10% of 100 nodes
        assert hasattr(data, 'train_mask')
        assert hasattr(data, 'val_mask')
        assert hasattr(data, 'test_mask')
    
    def test_load_dataset(self):
        """Test dataset loading."""
        # Test synthetic dataset
        data, dataset_type = load_dataset("synthetic_500_barabasi_albert_0.05")
        assert dataset_type == "anomaly_detection"
        assert data.num_nodes == 500
        assert hasattr(data, 'anomaly_labels')


class TestEvaluation:
    """Test evaluation metrics."""
    
    def setup_method(self):
        """Set up test fixtures."""
        set_seed(42)
        self.device = get_device()
        self.evaluator = AnomalyEvaluator(self.device)
        
        # Create dummy scores and labels
        self.scores = torch.randn(100)
        self.labels = torch.zeros(100, dtype=torch.bool)
        self.labels[torch.randperm(100)[:10]] = True  # 10 anomalies
    
    def test_evaluate(self):
        """Test evaluation function."""
        metrics = self.evaluator.evaluate(self.scores, self.labels)
        
        assert 'auroc' in metrics
        assert 'auprc' in metrics
        assert 'precision_at_10' in metrics
        assert 0 <= metrics['auroc'] <= 1
        assert 0 <= metrics['auprc'] <= 1
    
    def test_compute_anomaly_scores(self):
        """Test anomaly score computation."""
        # Create a simple model
        model = DOMINANT(16, 32, 16)
        x = torch.randn(50, 16)
        edge_index = torch.randint(0, 50, (2, 100))
        
        scores = compute_anomaly_scores(model, x, edge_index)
        assert scores.shape == (50,)
        assert not torch.isnan(scores).any()


class TestDevice:
    """Test device utilities."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cuda', 'mps', 'cpu']
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test that seeds are set
        import random
        assert random.getstate()[1][0] == 42
        
        # Test numpy seed
        np.random.seed(42)
        assert np.random.get_state()[1][0] == 42


if __name__ == "__main__":
    pytest.main([__file__])
