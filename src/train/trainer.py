"""Training utilities for anomaly detection models."""

import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as an improvement.
        restore_best_weights: Whether to restore best weights when stopping.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
    
    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        """Check if training should stop.
        
        Args:
            val_loss: Current validation loss.
            model: Model to potentially save weights for.
            
        Returns:
            True if training should stop, False otherwise.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
        
        return self.counter >= self.patience
    
    def restore_weights(self, model: nn.Module) -> None:
        """Restore best weights to model.
        
        Args:
            model: Model to restore weights to.
        """
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)


class AnomalyTrainer:
    """Trainer for anomaly detection models.
    
    Args:
        model: Model to train.
        device: Device for training.
        use_wandb: Whether to use Weights & Biases logging.
        project_name: W&B project name.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        use_wandb: bool = False,
        project_name: str = "gnn-anomaly-detection",
    ):
        self.model = model.to(device)
        self.device = device
        self.use_wandb = use_wandb
        
        if use_wandb:
            wandb.init(project=project_name)
            wandb.watch(model)
    
    def train_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        data: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> float:
        """Train for one epoch.
        
        Args:
            optimizer: Optimizer.
            data: Node features.
            edge_index: Edge indices.
            pos_edge_index: Positive edge indices.
            neg_edge_index: Negative edge indices.
            
        Returns:
            Training loss.
        """
        self.model.train()
        optimizer.zero_grad()
        
        if hasattr(self.model, 'compute_loss'):
            # DOMINANT model
            loss, struct_loss, attr_loss = self.model.compute_loss(
                data, edge_index, pos_edge_index, neg_edge_index
            )
        else:
            # GAE/VGAE models
            z = self.model.encode(data, edge_index)
            loss = self.model.recon_loss(z, pos_edge_index)
            
            if hasattr(self.model, 'kl_loss'):
                loss = loss + (1 / data.num_nodes) * self.model.kl_loss()
        
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    def validate(
        self,
        data: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
    ) -> float:
        """Validate the model.
        
        Args:
            data: Node features.
            edge_index: Edge indices.
            pos_edge_index: Positive edge indices.
            neg_edge_index: Negative edge indices.
            
        Returns:
            Validation loss.
        """
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'compute_loss'):
                loss, _, _ = self.model.compute_loss(
                    data, edge_index, pos_edge_index, neg_edge_index
                )
            else:
                z = self.model.encode(data, edge_index)
                loss = self.model.recon_loss(z, pos_edge_index)
                
                if hasattr(self.model, 'kl_loss'):
                    loss = loss + (1 / data.num_nodes) * self.model.kl_loss()
        
        return loss.item()
    
    def train(
        self,
        data: torch.Tensor,
        edge_index: torch.Tensor,
        pos_edge_index: torch.Tensor,
        neg_edge_index: torch.Tensor,
        epochs: int = 100,
        lr: float = 0.01,
        weight_decay: float = 5e-4,
        patience: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """Train the model.
        
        Args:
            data: Node features.
            edge_index: Edge indices.
            pos_edge_index: Positive edge indices.
            neg_edge_index: Negative edge indices.
            epochs: Number of training epochs.
            lr: Learning rate.
            weight_decay: Weight decay.
            patience: Early stopping patience.
            save_path: Path to save best model.
            
        Returns:
            Dictionary with training history.
        """
        # Move data to device
        data = data.to(self.device)
        edge_index = edge_index.to(self.device)
        pos_edge_index = pos_edge_index.to(self.device)
        neg_edge_index = neg_edge_index.to(self.device)
        
        # Setup optimizer
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Setup early stopping
        early_stopping = EarlyStopping(patience=patience)
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
        }
        
        # Training loop
        pbar = tqdm(range(epochs), desc="Training")
        for epoch in pbar:
            # Training
            train_loss = self.train_epoch(
                optimizer, data, edge_index, pos_edge_index, neg_edge_index
            )
            
            # Validation
            val_loss = self.validate(
                data, edge_index, pos_edge_index, neg_edge_index
            )
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            
            # Update progress bar
            pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'val_loss': f'{val_loss:.4f}',
            })
            
            # Log to W&B
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                })
            
            # Early stopping
            if early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Restore best weights
        early_stopping.restore_weights(self.model)
        
        # Save model
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(self.model.state_dict(), save_path)
            print(f"Model saved to {save_path}")
        
        return history


def train_model(
    model: nn.Module,
    data: torch.Tensor,
    edge_index: torch.Tensor,
    pos_edge_index: torch.Tensor,
    neg_edge_index: torch.Tensor,
    device: torch.device,
    config: Dict,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    """Train a model with given configuration.
    
    Args:
        model: Model to train.
        data: Node features.
        edge_index: Edge indices.
        pos_edge_index: Positive edge indices.
        neg_edge_index: Negative edge indices.
        device: Device for training.
        config: Training configuration.
        
    Returns:
        Tuple of (trained model, training history).
    """
    trainer = AnomalyTrainer(
        model=model,
        device=device,
        use_wandb=config.get('use_wandb', False),
        project_name=config.get('project_name', 'gnn-anomaly-detection'),
    )
    
    history = trainer.train(
        data=data,
        edge_index=edge_index,
        pos_edge_index=pos_edge_index,
        neg_edge_index=neg_edge_index,
        epochs=config.get('epochs', 100),
        lr=config.get('lr', 0.01),
        weight_decay=config.get('weight_decay', 5e-4),
        patience=config.get('patience', 10),
        save_path=config.get('save_path'),
    )
    
    return trainer.model, history
