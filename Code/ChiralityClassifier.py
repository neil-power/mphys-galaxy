from custom_models.G_ResNet_18 import G_ResNet18
from custom_models.Jia_ResNet import JiaResnet50
from custom_models.LeNet import VanillaLeNet
from custom_models.Steerable_LeNet import CNSteerableLeNet
import lightning as pl
import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchmetrics.functional.classification import multiclass_calibration_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from  matplotlib.colors import LinearSegmentedColormap

class ChiralityClassifier(pl.LightningModule):
    """
    Models:
    resnet18 - resnet34 - resnet50 - resnet101 - resnet152
    jiaresnet50 - LeNet
    G_ResNet18 - G_LeNet
    """
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
        batch_size=16,
        weights=None
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
        self.acc = self.accuracy_metric
        self.ece = self.ece_metric
        self.model = self.model_versions[model_version](num_classes=num_classes)
        if weights is not None:
            self.load_state_dict(torch.load(weights))
        self.test_y_predicted = []
        self.test_y_true = []

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
        
        preds = self(x)
        loss = self.loss_fn(preds, y)
        acc = self.acc(preds, y)
        ece = self.ece(preds, y)
        return loss, acc, ece

    def training_step(self, batch, batch_idx):
        loss, acc, ece = self._step(batch)

        self.log("train_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("calibration_error", ece, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, ece = self._step(batch)

        self.log("val_loss", loss, on_epoch=True, prog_bar=False, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("calibration_error", ece, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        loss, acc, ece = self._step(batch)

        x, y_true = batch
        y_preds = self(x)
        self.test_y_predicted.append(y_preds)
        self.test_y_true.append(y_true)

        self.log("test_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_acc", acc, on_epoch=True, prog_bar=True, logger=True)
        self.log("calibration_error", ece, on_epoch=True, prog_bar=True, logger=True)
        return y_true, y_preds

    def on_test_epoch_end(self):
         y_true = torch.argmax(torch.cat(self.test_y_true),dim=1)
         y_preds = torch.argmax(torch.cat(self.test_y_predicted),dim=1)
         cm = confusion_matrix(y_true, y_preds,normalize = 'true')

         disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['S','Z','None'])
         colours = ['#ffffcc','#a1dab4','#41b6c4','#2c7fb8','#253494']
         custom_cmap = LinearSegmentedColormap.from_list('custom_cmap', colours, N=20)
         disp.plot(xticks_rotation=45,cmap = custom_cmap,values_format=".3f",colorbar=False)
         for labels in disp.text_.ravel():
                labels.set_fontsize(26)
         ax = disp.ax_
         ax.set_xlabel('Predicted Labels',fontsize=12)
         ax.set_ylabel('True Labels',fontsize=12)
         ax.tick_params(axis='both', which='major', labelsize=12)
         ax.tick_params(axis='both', which='minor', labelsize=10)
         plt.savefig('confusion_test.png')

    def accuracy_metric(self,predicted_labels,true_labels):
        true_highest_prob = torch.argmax(true_labels, dim=1)
        predicted_highest_prob = torch.argmax(predicted_labels, dim=1)   

        matches = torch.count_nonzero(true_highest_prob==predicted_highest_prob)
        test_accuracy = matches/true_highest_prob.shape[0]
        return test_accuracy
    
    def ece_metric(self,predicted_labels,true_labels):
        metric = multiclass_calibration_error(predicted_labels, torch.argmax(true_labels, dim=1), num_classes=3, n_bins=3, norm='l1')
        return metric
