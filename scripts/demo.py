"""Example script demonstrating the modernized anomaly detection system."""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from src.utils.device import get_device, set_seed
from src.data.dataset import load_dataset, get_data_info
from src.models.anomaly_models import create_model
from src.train.trainer import train_model
from src.eval.metrics import AnomalyEvaluator, compute_anomaly_scores


def main():
    """Demonstrate the modernized anomaly detection system."""
    print("🔍 Graph Neural Networks for Anomaly Detection")
    print("=" * 50)
    
    # Set up device and seeding
    device = get_device()
    set_seed(42)
    print(f"Using device: {device}")
    
    # Load dataset
    print("\n📊 Loading dataset...")
    data, dataset_type = load_dataset(
        dataset_name="synthetic_1000_barabasi_albert_0.1",
        root="./data",
    )
    
    # Print dataset information
    data_info = get_data_info(data)
    print(f"Dataset info: {data_info}")
    
    # Create model
    print("\n🤖 Creating model...")
    model = create_model(
        model_name="gae",
        in_channels=data.num_node_features,
        hidden_channels=64,
        out_channels=32,
    )
    
    print(f"Model: GAE")
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Prepare training data
    if hasattr(data, 'train_pos_edge_index'):
        train_pos_edge_index = data.train_pos_edge_index
        train_neg_edge_index = data.train_neg_edge_index
        edge_index = data.train_pos_edge_index
    else:
        from torch_geometric.utils import train_test_split_edges
        data = train_test_split_edges(data)
        train_pos_edge_index = data.train_pos_edge_index
        train_neg_edge_index = data.train_neg_edge_index
        edge_index = data.train_pos_edge_index
    
    # Training configuration
    train_config = {
        'epochs': 50,  # Reduced for demo
        'lr': 0.01,
        'weight_decay': 5e-4,
        'patience': 10,
        'use_wandb': False,
        'project_name': 'gnn-anomaly-detection-demo',
        'save_path': None,  # Don't save for demo
    }
    
    # Train model
    print("\n🚀 Starting training...")
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
    print("\n📈 Evaluating model...")
    evaluator = AnomalyEvaluator(device=device)
    
    if hasattr(data, 'anomaly_labels'):
        # Compute anomaly scores
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
            k_values=[10, 50, 100],
        )
        
        print("\n🎯 Evaluation Results:")
        print(f"AUROC: {metrics['auroc']:.4f}")
        print(f"AUPRC: {metrics['auprc']:.4f}")
        print(f"Precision@10: {metrics['precision_at_10']:.4f}")
        print(f"Precision@50: {metrics['precision_at_50']:.4f}")
        print(f"Precision@100: {metrics['precision_at_100']:.4f}")
        
        # Show top anomalous nodes
        print("\n🔍 Top 10 Most Anomalous Nodes:")
        top_indices = scores.topk(10).indices
        top_scores = scores[top_indices]
        
        for i, (node_id, score) in enumerate(zip(top_indices, top_scores)):
            is_anomaly = data.anomaly_labels[node_id].item()
            status = "✓ Anomaly" if is_anomaly else "✗ Normal"
            print(f"{i+1:2d}. Node {node_id:4d}: {score:.4f} ({status})")
        
        # Calculate detection accuracy
        detected_anomalies = sum(data.anomaly_labels[node_id].item() for node_id in top_indices)
        detection_rate = detected_anomalies / 10
        print(f"\nDetection Rate (Top 10): {detection_rate:.1%}")
        
    else:
        print("No anomaly labels found for evaluation")
    
    print("\n✅ Demo completed successfully!")
    print("\nTo run the interactive demo:")
    print("streamlit run demo/app.py")


if __name__ == "__main__":
    main()
