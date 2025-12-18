"""Evaluation metrics for anomaly detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    precision_at_k,
)
from torchmetrics import AUROC, AveragePrecision


class AnomalyEvaluator:
    """Evaluator for anomaly detection models.
    
    Computes various metrics including AUROC, AUPRC, Precision@K, and
    provides detailed analysis of model performance.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """Initialize evaluator.
        
        Args:
            device: Device for computations.
        """
        self.device = device or torch.device("cpu")
        self.auroc_metric = AUROC(task="binary").to(self.device)
        self.ap_metric = AveragePrecision(task="binary").to(self.device)
    
    def evaluate(
        self,
        scores: torch.Tensor,
        labels: torch.Tensor,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Evaluate anomaly detection performance.
        
        Args:
            scores: Anomaly scores (higher = more anomalous).
            labels: Ground truth labels (1 = anomalous, 0 = normal).
            k_values: List of k values for Precision@K.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if k_values is None:
            k_values = [10, 50, 100, 200]
        
        # Convert to numpy for sklearn metrics
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        # Basic metrics
        auroc = roc_auc_score(labels_np, scores_np)
        auprc = average_precision_score(labels_np, scores_np)
        
        # Precision@K metrics
        precision_at_k_metrics = {}
        for k in k_values:
            if k <= len(scores_np):
                precision_at_k_metrics[f"precision_at_{k}"] = precision_at_k(
                    labels_np, scores_np, k
                )
        
        # Additional analysis
        metrics = {
            "auroc": auroc,
            "auprc": auprc,
            **precision_at_k_metrics,
        }
        
        # Add detailed analysis
        metrics.update(self._detailed_analysis(scores_np, labels_np))
        
        return metrics
    
    def _detailed_analysis(
        self, scores: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Perform detailed analysis of model performance.
        
        Args:
            scores: Anomaly scores.
            labels: Ground truth labels.
            
        Returns:
            Dictionary with detailed metrics.
        """
        # Score statistics
        normal_scores = scores[labels == 0]
        anomaly_scores = scores[labels == 1]
        
        metrics = {
            "normal_score_mean": normal_scores.mean(),
            "normal_score_std": normal_scores.std(),
            "anomaly_score_mean": anomaly_scores.mean(),
            "anomaly_score_std": anomaly_scores.std(),
            "score_separation": anomaly_scores.mean() - normal_scores.mean(),
        }
        
        # Threshold analysis
        fpr, tpr, thresholds = roc_curve(labels, scores)
        optimal_idx = np.argmax(tpr - fpr)
        optimal_threshold = thresholds[optimal_idx]
        
        metrics.update({
            "optimal_threshold": optimal_threshold,
            "optimal_tpr": tpr[optimal_idx],
            "optimal_fpr": fpr[optimal_idx],
        })
        
        return metrics
    
    def evaluate_model(
        self,
        model: torch.nn.Module,
        data: torch.Tensor,
        edge_index: torch.Tensor,
        labels: torch.Tensor,
        k_values: Optional[List[int]] = None,
    ) -> Dict[str, float]:
        """Evaluate a model directly.
        
        Args:
            model: Anomaly detection model.
            data: Node features.
            edge_index: Edge indices.
            labels: Ground truth labels.
            k_values: List of k values for Precision@K.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        model.eval()
        with torch.no_grad():
            if hasattr(model, "anomaly_score"):
                scores = model.anomaly_score(data, edge_index)
            elif hasattr(model, 'encode'):
                # For GAE/VGAE models
                z = model.encode(data, edge_index)
                if hasattr(model, 'recon_loss'):
                    # Compute reconstruction error for each node
                    scores = torch.zeros(data.size(0), device=data.device)
                    for i in range(data.size(0)):
                        # Approximate reconstruction error for node i
                        node_edges = edge_index[:, edge_index[0] == i]
                        if node_edges.size(1) > 0:
                            recon_loss = model.recon_loss(z, node_edges)
                            scores[i] = recon_loss
                else:
                    # Fallback: use embedding norm
                    scores = torch.norm(z, dim=1)
            else:
                raise ValueError("Model must have anomaly_score or encode method")
        
        return self.evaluate(scores, labels, k_values)


def compute_anomaly_scores(
    model: torch.nn.Module,
    data: torch.Tensor,
    edge_index: torch.Tensor,
    method: str = "reconstruction",
) -> torch.Tensor:
    """Compute anomaly scores using different methods.
    
    Args:
        model: Trained model.
        data: Node features.
        edge_index: Edge indices.
        method: Method for computing scores ('reconstruction', 'embedding_norm', 'distance').
        
    Returns:
        Anomaly scores for each node.
    """
    model.eval()
    with torch.no_grad():
        if hasattr(model, 'encode'):
            z = model.encode(data, edge_index)
            
            if method == "reconstruction":
                # Compute reconstruction error
                if hasattr(model, 'recon_loss'):
                    scores = torch.zeros(data.size(0), device=data.device)
                    for i in range(data.size(0)):
                        node_edges = edge_index[:, edge_index[0] == i]
                        if node_edges.size(1) > 0:
                            recon_loss = model.recon_loss(z, node_edges)
                            scores[i] = recon_loss
                    return scores
                else:
                    # Approximate reconstruction error
                    recon_adj = torch.sigmoid(torch.matmul(z, z.t()))
                    scores = ((recon_adj - torch.eye(z.size(0), device=z.device)) ** 2).sum(dim=1)
                    return scores
            
            elif method == "embedding_norm":
                # Use embedding norm as anomaly score
                return torch.norm(z, dim=1)
            
            elif method == "distance":
                # Distance from centroid
                centroid = z.mean(dim=0)
                distances = torch.norm(z - centroid, dim=1)
                return distances
            
            else:
                raise ValueError(f"Unknown method: {method}")
        
        elif hasattr(model, 'anomaly_score'):
            return model.anomaly_score(data, edge_index)
        
        else:
            raise ValueError("Model must have encode or anomaly_score method")


def create_evaluation_report(
    metrics: Dict[str, float],
    model_name: str,
    dataset_name: str,
) -> str:
    """Create a formatted evaluation report.
    
    Args:
        metrics: Dictionary of evaluation metrics.
        model_name: Name of the model.
        dataset_name: Name of the dataset.
        
    Returns:
        Formatted report string.
    """
    report = f"""
# Evaluation Report

## Model: {model_name}
## Dataset: {dataset_name}

## Performance Metrics
- **AUROC**: {metrics['auroc']:.4f}
- **AUPRC**: {metrics['auprc']:.4f}

## Precision@K
"""
    
    for key, value in metrics.items():
        if key.startswith('precision_at_'):
            k = key.split('_')[-1]
            report += f"- **Precision@{k}**: {value:.4f}\n"
    
    report += f"""
## Score Analysis
- **Normal Score Mean**: {metrics['normal_score_mean']:.4f}
- **Normal Score Std**: {metrics['normal_score_std']:.4f}
- **Anomaly Score Mean**: {metrics['anomaly_score_mean']:.4f}
- **Anomaly Score Std**: {metrics['anomaly_score_std']:.4f}
- **Score Separation**: {metrics['score_separation']:.4f}

## Optimal Threshold Analysis
- **Optimal Threshold**: {metrics['optimal_threshold']:.4f}
- **True Positive Rate**: {metrics['optimal_tpr']:.4f}
- **False Positive Rate**: {metrics['optimal_fpr']:.4f}
"""
    
    return report
