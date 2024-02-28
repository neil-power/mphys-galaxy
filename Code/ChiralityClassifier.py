from custom_models.G_ResNet_18 import G_ResNet18
from custom_models.Jia_ResNet import JiaResnet50
from custom_models.LeNet import VanillaLeNet
from custom_models.Steerable_LeNet import CNSteerableLeNet
import lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torcheval.metrics import BinaryAccuracy

class ChiralityClassifier(pl.LightningModule):
    model_versions = {
        "resnet18": models.resnet18,
        "resnet34": models.resnet34,
        "resnet50": models.resnet50,
        "resnet101": models.resnet101,
        "resnet152": models.resnet152,
        "jiaresnet50": JiaResnet50,
        "G_ResNet18": G_ResNet18,
        "LeNet": VanillaLeNet,
        "G_LeNet": CNSteerableLeNet
    }
    optimizers = {"adamw": optim.AdamW, "sgd": optim.SGD}
    schedulers = {"steplr": optim.lr_scheduler.StepLR}

    def __init__(
        self,
        model_version,
        num_classes=3,
        optimizer="adamw",
        scheduler  ="steplr",
        lr=1e-3,
        weight_decay=0,
        step_size=5,
        gamma=0.85,
        batch_size=16
    ):
        super().__init__()

        self.num_classes = num_classes
        self.lr = lr
        self.weight_decay = weight_decay
        self.step_size = step_size
        self.gamma = gamma
        self.batch_size = batch_size
        self.optimizer = self.optimizers[optimizer]
        self.scheduler = self.schedulers[scheduler]
        self.loss_fn = nn.CrossEntropyLoss()
        self.acc = self.accuracy_metric #Accuracy(task="multiclass", num_classes=num_classes)
        self.model = self.model_versions[model_version](num_classes=num_classes)

    def forward(self, X):
        return self.model(X)

    def configure_optimizers(self):
        optimizer_class = self.optimizer(self.parameters(), lr=self.lr,weight_decay=self.weight_decay)
        scheduler = self.scheduler(optimizer_class, step_size=self.step_size, gamma=self.gamma)
        return {
        "optimizer": optimizer_class,
        "lr_scheduler": {"scheduler": scheduler},
        }

    def _step(self, batch):
        x, y = batch
        
        if self.num_classes == 2: # NOT IDEAL, CATCH FOR JIARESNET
            preds = self.predict(x)
        else:
            preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        return loss, acc

    def training_step(self, batch, batch_idx):
        #time here
        loss, acc = self._step(batch)
        # perform logging
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc = self._step(batch)
        # perform logging
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_step=True, prog_bar=True, logger=True)

    def accuracy_metric(self,predicted_labels,true_labels):
        #Takes in softmaxed labels, checks if max column is the same

        true_highest_prob = torch.argmax(true_labels, dim=1)
        predicted_highest_prob = torch.argmax(predicted_labels, dim=1)   
        
        metric = BinaryAccuracy()
        metric.update(predicted_highest_prob,true_highest_prob)
        test_accuracy = metric.compute()
        return test_accuracy