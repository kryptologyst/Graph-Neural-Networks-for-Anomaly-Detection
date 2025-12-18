"""Main training script for anomaly detection experiments."""

import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any

import torch
import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.device import get_device, set_seed
from src.data.dataset import load_dataset, get_data_info
from src.models.anomaly_models import create_model
from src.train.trainer import train_model
from src.eval.metrics import AnomalyEvaluator, compute_anomaly_scores


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main training function.
    
    Args:
        cfg: Hydra configuration object.
    """
    # Set up device and seeding
    device = get_device()
    set_seed(cfg.device.seed)
    
    print(f"Using device: {device}")
    print(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # Load dataset
    print("Loading dataset...")
    data, dataset_type = load_dataset(
        dataset_name=cfg.dataset.name,
        root=cfg.paths.data_dir,
        anomaly_ratio=cfg.dataset.anomaly_ratio,
        graph_type=cfg.dataset.graph_type,
        anomaly_type=cfg.dataset.anomaly_type,
    )
    
    # Print dataset info
    data_info = get_data_info(data)
    print(f"Dataset info: {data_info}")
    
    # Create model
    print("Creating model...")
    model = create_model(
        model_name=cfg.model.name,
        in_channels=data.num_node_features,
        hidden_channels=cfg.model.hidden_channels,
        out_channels=cfg.model.out_channels,
        dropout=cfg.model.dropout,
        num_layers=cfg.model.num_layers,
        num_heads=cfg.model.num_heads,
    )
    
    print(f"Model: {cfg.model.name}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Prepare training data
    if hasattr(data, 'train_pos_edge_index'):
        # For link prediction datasets
        train_pos_edge_index = data.train_pos_edge_index
        train_neg_edge_index = data.train_neg_edge_index
        edge_index = data.train_pos_edge_index
    else:
        # For synthetic datasets, create edge splits
        from torch_geometric.utils import train_test_split_edges
        data = train_test_split_edges(data)
        train_pos_edge_index = data.train_pos_edge_index
        train_neg_edge_index = data.train_neg_edge_index
        edge_index = data.train_pos_edge_index
    
    # Training configuration
    train_config = {
        'epochs': cfg.training.epochs,
        'lr': cfg.training.lr,
        'weight_decay': cfg.training.weight_decay,
        'patience': cfg.training.patience,
        'use_wandb': cfg.training.use_wandb,
        'project_name': cfg.training.project_name,
        'save_path': os.path.join(cfg.paths.checkpoints_dir, f"{cfg.model.name}_best.pth"),
    }
    
    # Train model
    print("Starting training...")
    trained_model, history = train_model(
        model=model,
        data=data.x,
        edge_index=edge_index,
        pos_edge_index=train_pos_edge_index,
        neg_edge_index=train_neg_edge_index,
        device=device,
        config=train_config,
    )
    
    # Evaluate model
    print("Evaluating model...")
    evaluator = AnomalyEvaluator(device=device)
    
    # Get anomaly scores
    if hasattr(data, 'anomaly_labels'):
        scores = compute_anomaly_scores(
            model=trained_model,
            data=data.x,
            edge_index=edge_index,
            method="reconstruction",
        )
        
        # Evaluate performance
        metrics = evaluator.evaluate(
            scores=scores,
            labels=data.anomaly_labels,
            k_values=cfg.evaluation.k_values,
        )
        
        print("\nEvaluation Results:")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        
        for k in cfg.evaluation.k_values:
            if f'precision_at_{k}' in metrics:
                print(f"Precision@{k}: {metrics[f'precision_at_{k}']:.4f}")
        
        # Save results
        os.makedirs(cfg.paths.assets_dir, exist_ok=True)
        results_path = os.path.join(cfg.paths.assets_dir, f"{cfg.model.name}_results.txt")
        
        with open(results_path, 'w') as f:
            f.write(f"Model: {cfg.model.name}\n")
            f.write(f"Dataset: {cfg.dataset.name}\n")
            f.write(f"AUROC: {metrics['auroc']:.4f}\n")
            f.write(f"AUPRC: {metrics['auprc']:.4f}\n")
            for k in cfg.evaluation.k_values:
                if f'precision_at_{k}' in metrics:
                    f.write(f"Precision@{k}: {metrics[f'precision_at_{k}']:.4f}\n")
        
        print(f"Results saved to {results_path}")
    
    else:
        print("No anomaly labels found for evaluation")
    
    print("Training completed!")


if __name__ == "__main__":
    main()
