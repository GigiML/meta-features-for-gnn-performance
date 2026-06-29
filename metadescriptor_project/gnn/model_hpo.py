import torch.nn as nn
import lightning as L
import torch
import torch.optim as optim
from torch.nn import Module
from torchmetrics import MetricCollection
from torchmetrics import Accuracy
from torchmetrics.classification import AveragePrecision, AUROC, F1Score
from src.gnn.model import GNN, GIN, GAT
from torch_geometric.nn import summary
from torch_geometric.data import Data   


GNN_dict = {
    "GCN": GNN,
    "GIN": GIN,
    "GAT": GAT,
}


class GraphLitModel(L.LightningModule):
    def __init__(
        self,
        # architecture
        net: str,
        num_layers: int = 2,
        hidden_channels: int = 128,
        in_channels: int = 10,
        out_channels: int = 2,
        criterion: Module = nn.CrossEntropyLoss, 
        # optimize parameter
        lr: float = 1e-3,
        beta1: float = 0.9,
        weight_decay: float = 0,
        normalize: bool = True,
        weights:torch.tensor = torch.tensor((0.5,0.5))
    ) -> None:
        super().__init__()
        if normalize:
            self.net = GNN_dict[net](
                in_channels=in_channels,
                num_layers=num_layers,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                normalize=normalize,
            )
        else:
            self.net = GNN_dict[net](
                in_channels=in_channels,
                num_layers=num_layers,
                hidden_channels=hidden_channels,
                out_channels=out_channels,
                normalize=normalize,
            )
        self.criterion = criterion
        self.weights = weights

        # ── class metric ────────────────────────────────────────
        class_metrics = {
            "Balanced_Acc": Accuracy(
                task="multiclass", num_classes=out_channels, average="macro"
            ),
            "Accuracy": Accuracy(
                task="multiclass", num_classes=out_channels, average="micro"
            ),
            "F1": F1Score(task="multiclass", num_classes=out_channels, average="micro"),
        }

        self.train_class_metrics = MetricCollection(class_metrics, prefix="train/")
        self.val_class_metrics = MetricCollection(class_metrics, prefix="val/")
        self.test_class_metrics = MetricCollection(class_metrics, prefix="test/")

        # ── Proba metrcic ────────────────────────────────────────
        proba_metrics = {
            "AUROC": AUROC(task="multiclass", num_classes=out_channels),
            "AUPRC": AveragePrecision(task="multiclass", num_classes=out_channels),
        }

        self.train_proba_metrics = MetricCollection(proba_metrics, prefix="train/")
        self.val_proba_metrics = MetricCollection(proba_metrics, prefix="val/")
        self.test_proba_metrics = MetricCollection(proba_metrics, prefix="test/")

        # ── loss function ────────────────────────────────────────
        self.register_buffer("class_weights", weights.float())
        self.criterion_train = criterion(weight=self.class_weights)
        self.criterion_train = criterion(weight=self.class_weights)
        self.criterion_test = criterion(weight=self.class_weights)
        self.criterion_val = criterion(weight=self.class_weights)

        self.save_hyperparameters()

    def on_fit_start(self):
            """Log GNN summary dans TensorBoard"""
            if 0 and self.global_rank == 0 and self.logger:
                x = torch.randn(64, self.hparams.in_channels)
                edge_index = torch.randint(0, 64, (2, 128))
                batch = torch.zeros(64, dtype=torch.long)  
                gnn_summary = summary(self.net, x, edge_index, batch)
                selear.logger.experiment.add_text("GNN_Summary", str(gnn_summary))
                
    def training_step(self, batch):
        logits = self.net(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion_train(logits, batch.y)
        self.log("train/loss", loss, on_epoch=True)
        preds = logits.argmax(dim=1)
        self.train_class_metrics.update(preds, batch.y)
        self.train_proba_metrics.update(logits, batch.y)
        self.log_dict(self.train_class_metrics, sync_dist=True)
        self.log_dict(self.train_proba_metrics, sync_dist=True)
        return loss

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch.to(device)


    def test_step(self, batch, batch_idx):
        logits = self.net(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion_test(logits, batch.y)
        self.log("test/loss", loss, on_epoch=True)
        preds = logits.argmax(dim=1)
        self.test_class_metrics.update(preds, batch.y)
        self.test_proba_metrics.update(logits, batch.y)
        self.log_dict(self.test_class_metrics, sync_dist=True)
        self.log_dict(self.test_proba_metrics, sync_dist=True)

    def validation_step(self, batch):
        logits = self.net(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion_val(logits, batch.y)
        self.log("val/loss", loss, on_epoch=True)
        preds = logits.argmax(dim=1)
        self.val_class_metrics.update(preds, batch.y)
        self.val_proba_metrics.update(logits, batch.y)

    def on_validation_epoch_end(self):
        self.log_dict(self.val_class_metrics, sync_dist=True)
        self.log_dict(self.val_proba_metrics, sync_dist=True)

    def configure_optimizers(self):
        return optim.Adam(
            self.net.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
        )

    
