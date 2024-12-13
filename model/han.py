"""
This module defines the `BNNHAN` model and the `BNNHANPOC` class for implementing
and testing a Heterogeneous Graph Attention Network (HAN) on EEG-based chronic pain detection data.

- `BNNHAN` is a neural network model using HANConv layers suitable for heterogeneous graphs.
- `BNNHANPOC` provides a proof-of-concept implementation, including training, evaluation,
  and performance reporting functionalities.

The module leverages PyTorch and PyTorch Geometric for handling graph data and neural network operations.
"""

from typing import Optional, Dict, Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import Linear, HANConv
from torch_geometric.typing import Metadata, Adj, EdgeType
from torch_geometric.data import HeteroData

from sklearn.metrics import classification_report

import config.includes as inc
from model.dataset import BNNHDataSet

# Set random seeds for reproducibility
torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BNNHAN(nn.Module):
    """Heterogeneous Graph Attention Network (HAN) model for EEG-based chronic pain detection."""

    DROPOUT: float = 0.6  # Dropout rate for regularization

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_h: int = 128,
        heads: int = 8,
        metadata: Optional[Metadata] = None,
    ) -> None:
        """
        Initialize the BNNHAN model.

        Args:
            dim_in (int): Input feature dimension.
                - **Important**: Use `-1` for automatic inference of input dimensions per node type.
            dim_out (int): Output feature dimension (number of classes).
            dim_h (int, optional): Hidden layer dimension. Defaults to 128.
            heads (int, optional): Number of attention heads. Defaults to 8.
            metadata (Optional[Metadata], optional): Metadata of the heterogeneous graph. Defaults to None.
        """
        super().__init__()
        metadata = metadata or ([], [])
        # HAN convolutional layer
        self.han = HANConv(
            in_channels=dim_in,
            out_channels=dim_h,
            heads=heads,
            dropout=self.DROPOUT,
            metadata=metadata,
        )
        # Linear layer for classification
        self.linear = Linear(dim_h, dim_out)

    def forward(
        self,
        x_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Tensor:
        """
        Forward pass of the model.

        Args:
            x_dict (Dict[str, Tensor]): Dictionary of node feature tensors for each node type.
            edge_index_dict (Dict[EdgeType, Adj]): Dictionary of edge index tensors for each edge type.

        Returns:
            Tensor: Output logits for classification.
        """
        # Apply HAN convolution
        out = self.han(x_dict, edge_index_dict)
        # Retrieve the output for SUBJECT node type
        subject_out = out["SUBJECT"]
        # Apply linear layer to obtain final output
        out = self.linear(subject_out)
        return out


class BNNHANPOC:
    """Proof-of-Concept (POC) class for training and evaluating the BNNHAN model."""

    def __init__(self) -> None:
        """
        Initialize the BNNHANPOC.

        Loads the dataset, applies necessary transformations, and prepares the data for training.
        """
        self.path: str = inc.BNNHDSDIR

        # Define metapaths for the HAN model
        metapaths = [
            [("SUBJECT", "READ_LOC"), ("READ_LOC", "WAVE_ABP")],
            [("WAVE_ABP", "READ_LOC"), ("READ_LOC", "SUBJECT")],
        ]

        # Transformation to add metapaths to the data
        transform = T.AddMetaPaths(
            metapaths=metapaths,
            drop_orig_edge_types=True,
            drop_unconnected_node_types=True,
        )

        # Load dataset with the specified transformation
        dataset = BNNHDataSet(root=self.path, transform=transform)
        self.data: HeteroData = dataset[0]

    @torch.no_grad()
    def test(self, mask: Tensor, model: nn.Module, data: HeteroData) -> float:
        """
        Evaluate the model on the given mask (train, validation, or test set).

        Args:
            mask (Tensor): Boolean tensor indicating which nodes to include.
            model (nn.Module): The trained model to evaluate.
            data (HeteroData): The data containing node features and edge indices.

        Returns:
            float: Accuracy score on the specified nodes.
        """
        model.eval()
        # Get model predictions
        out = model(data.x_dict, data.edge_index_dict)
        pred = out.argmax(dim=-1)
        # Calculate accuracy
        correct = (pred[mask] == data["SUBJECT"].y[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / total if total > 0 else 0
        return acc

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: HeteroData,
        epochs: int = 200,
        patience: int = 100,
    ) -> None:
        """
        Train the model with early stopping.

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): Optimizer for updating model parameters.
            data (HeteroData): The data containing node features and edge indices.
            epochs (int, optional): Maximum number of epochs to train. Defaults to 200.
            patience (int, optional): Early stopping patience. Defaults to 100.
        """
        best_val_acc: float = 0.0
        patience_counter: int = patience

        for epoch in range(1, epochs + 1):
            model.train()
            optimizer.zero_grad()
            # Forward pass
            out = model(data.x_dict, data.edge_index_dict)
            # Compute loss on training data
            train_mask = data["SUBJECT"].train_mask
            val_mask = data["SUBJECT"].val_mask
            test_mask = data["SUBJECT"].test_mask
            loss = F.cross_entropy(out[train_mask], data["SUBJECT"].y[train_mask])
            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Evaluate on validation set
            val_acc = self.test(val_mask, model, data)

            # Check for improvement
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = patience  # Reset patience counter
            else:
                patience_counter -= 1  # Decrease patience counter

            # Print progress every 20 epochs
            if epoch % 20 == 0 or epoch == 1:
                train_acc = self.test(train_mask, model, data)
                test_acc = self.test(test_mask, model, data)
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                    f"Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}, "
                    f"Patience: {patience_counter}"
                )

            # Early stopping condition
            if patience_counter <= 0:
                print(f"Early stopping at epoch {epoch} due to no improvement.")
                break

    @torch.no_grad()
    def performance_report(
        self, mask: Tensor, model: nn.Module, data: HeteroData
    ) -> None:
        """
        Generate a detailed classification report.

        Args:
            mask (Tensor): Boolean tensor indicating which nodes to include.
            model (nn.Module): The trained model.
            data (HeteroData): The data containing node features and labels.
        """
        model.eval()
        # Get model predictions
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
        y_true = data["SUBJECT"].y[mask].cpu()
        y_pred = pred[mask].cpu()
        # Define label names
        labels = ["Chronic-Pain", "Control"]
        # Print classification report
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))

    def run(self) -> None:
        """
        Execute the full training and evaluation pipeline.
        """
        # Initialize the model with appropriate dimensions
        # Important: Use dim_in = -1 to let HANConv automatically infer input feature dimensions for each node type.
        # Do not set dim_in to self.data["SUBJECT"].num_node_features, as node types may have different input dimensions.
        model = BNNHAN(
            dim_in=-1,
            dim_out=2,
            metadata=self.data.metadata(),
        )
        # Define the optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        # Set the device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Move data and model to device
        data, model = self.data.to(device), model.to(device)
        # Start training
        self.fit(model, optimizer, data)
        # Evaluate on test set
        test_acc = self.test(data["SUBJECT"].test_mask, model, data)
        print(f"Test Accuracy: {test_acc*100:.2f}%")
        # Generate performance report
        self.performance_report(data["SUBJECT"].test_mask, model, data)
        print("Training and evaluation completed.")
