from typing import Optional, Dict

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

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.cuda.manual_seed_all(1)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class BNNHAN(nn.Module):
    """HAN model for BNN EEG graph heterogeneous data."""

    DROPOUT = 0.6

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        dim_h: int = 128,
        heads: int = 8,
        metadata: Optional[Metadata] = None,
    ) -> None:
        """Initialize the BNNHAN model.

        Args:
            dim_in (int): Input feature dimension.
            dim_out (int): Output feature dimension.
            dim_h (int, optional): Hidden layer dimension. Defaults to 128.
            heads (int, optional): Number of attention heads. Defaults to 8.
            metadata (Metadata, optional): Metadata for the graph. Defaults to None.
        """

        metadata = metadata or ([], [])

        super().__init__()
        self.han = HANConv(
            dim_in, dim_h, heads=heads, dropout=self.DROPOUT, metadata=metadata
        )
        self.linear = Linear(dim_h, dim_out)

    def forward(
        self, x_dict: Dict[str, Tensor], edge_index_dict: Dict[EdgeType, Adj]
    ) -> Tensor:
        """Forward pass of the model.

        Args:
            x_dict (Dict[str, Tensor]): Node feature dictionary.
            edge_index_dict (Dict[EdgeType, Adj]): Edge index dictionary.

        Returns:
            Tensor: Output tensor after the forward pass.
        """
        out = self.han.forward(x_dict, edge_index_dict)
        out = self.linear.forward(out["SUBJECT"])
        return out


class BNNHANPOC:
    """HAN proof of concept."""

    def __init__(self) -> None:
        """Initialize the BNNHAN proof of concept."""
        self.path = inc.BNNHDSDIR

        metapaths = [
            [("SUBJECT", "READ_LOC"), ("READ_LOC", "WAVE_ABP")],
            [("WAVE_ABP", "READ_LOC"), ("READ_LOC", "SUBJECT")],
        ]

        transform = T.AddMetaPaths(
            metapaths=metapaths,
            drop_orig_edge_types=True,  # type: ignore
            drop_unconnected_node_types=True,
        )

        dataset = BNNHDataSet(root=self.path, transform=transform)

        self.data = dataset[0]

    @torch.no_grad()
    def test(self, mask: Tensor, model: nn.Module, data: HeteroData) -> float:
        """Evaluate the model on the given mask.

        Args:
            mask (Tensor): Mask for selecting data.
            model (nn.Module): The model to evaluate.
            data (HeteroData): The data to evaluate on.

        Returns:
            float: Accuracy on the given mask.
        """
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
        acc = (pred[mask] == data["SUBJECT"].y[mask]).sum() / mask.sum()
        return float(acc)

    def fit(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data: HeteroData,
        epochs: int = 200,
        patience: int = 100,
    ) -> None:
        """Train the model.

        Args:
            model (nn.Module): The model to train.
            optimizer (torch.optim.Optimizer): The optimizer for training.
            data (HeteroData): The data to train on.
            epochs (int, optional): Number of epochs. Defaults to 200.
            patience (int, optional): Early stopping patience. Defaults to 100.
        """
        best_val_acc = 0.0
        start_patience = patience

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            out = model(data.x_dict, data.edge_index_dict)
            mask = data["SUBJECT"].train_mask
            loss = F.cross_entropy(out[mask], data["SUBJECT"].y[mask])
            loss.backward()
            optimizer.step()

            if epoch % 20 == 0:
                train_acc = self.test(data["SUBJECT"].train_mask, model, data)
                val_acc = self.test(data["SUBJECT"].val_mask, model, data)
                test_acc = self.test(data["SUBJECT"].test_mask, model, data)
                print(
                    f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, "
                    f"Val: {val_acc:.4f}, Test: {test_acc:.4f}"
                )

                if best_val_acc <= val_acc:
                    patience = start_patience
                    best_val_acc = val_acc
                else:
                    patience -= 1

            if patience <= 0:
                print(
                    "Stopping training as validation accuracy did not improve "
                    f"for {start_patience} epochs"
                )
                break

    @torch.no_grad()
    def performance_report(
        self, mask: Tensor, model: nn.Module, data: HeteroData
    ) -> None:
        """Generate a performance report.

        Args:
            mask (Tensor): Mask for selecting data.
            model (nn.Module): The model to evaluate.
            data (HeteroData): The data to evaluate on.
        """
        model.eval()
        pred = model(data.x_dict, data.edge_index_dict).argmax(dim=-1)
        y_true = data["SUBJECT"].y[mask].cpu()
        y_pred = pred[mask].cpu()
        labels = ["Chronic-Pain", "Control"]
        print(classification_report(y_true, y_pred, target_names=labels, digits=4))
        return

    def run(self) -> None:
        """Run the proof of concept."""
        model = BNNHAN(dim_in=-1, dim_out=2, metadata=self.data.metadata())
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        data, model = self.data.to(str(device)), model.to(str(device))

        print("-" * 40, "\n")
        self.fit(model, optimizer, data)
        test_acc = self.test(data["SUBJECT"].test_mask, model, data)
        print(f"Test accuracy: {test_acc*100:.2f}%")
        print("-" * 40, "\n")
        self.performance_report(data["SUBJECT"].test_mask, model, data)
        print("-" * 40, "\n")
        return
