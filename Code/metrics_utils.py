## Utilities used for processing, exporting and viewing model metrics

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard_reducer.event_loader import EventAccumulator
from dataset_utils import create_folder


# ---------------------------------------------------------------------------------

def save_metrics_from_logger(model_id,log_path,metrics_path,version=0,mode='train',save=True):
    metrics = pd.read_csv(f"{log_path}/{model_id}/version_{version}_{mode}/metrics.csv")
    if mode == 'train':
        metrics = metrics.drop(['train_loss_step',	'train_acc_step',	'train_calibration_error_step'],axis=1)
    metrics = metrics.groupby(metrics['epoch']).first()
    if save:
        save_dir = f"{metrics_path}/{model_id}/version_{version}"
        create_folder(save_dir)
        metrics.to_csv(f"{save_dir}/{mode}_metrics.csv")
    return metrics

# ---------------------------------------------------------------------------------

def get_metrics_from_csv(model_id,metrics_path,version=0,mode='train'):
    metrics = pd.read_csv(f"{metrics_path}/{model_id}/version_{version}/{mode}_metrics.csv")
    return metrics

# ---------------------------------------------------------------------------------

def plot_train_metrics(model_id,metrics_path,version=0,show=False,save=True):
    metrics = get_metrics_from_csv(model_id,metrics_path,version,mode='train')
    train_loss = metrics['train_loss_epoch']
    train_acc = metrics['train_acc_epoch']
    val_loss = metrics['val_loss']
    val_acc = metrics['val_acc']
    epoch = metrics.index

    fig = plt.figure(figsize=(9,4))
    ax1 = fig.add_subplot(221) 
    ax1.plot(epoch,train_loss,linestyle='-', c='orange')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Train Loss')

    ax2 = fig.add_subplot(222) 
    ax2.plot(epoch,val_loss,linestyle='-', c='violet')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Validation Loss')

    ax3 = fig.add_subplot(223) 
    ax3.plot(epoch,train_acc,linestyle='-', c='red')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Train Accuracy')

    ax4 = fig.add_subplot(224) 
    ax4.plot(epoch,val_acc,linestyle='-', c='green')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Validation Accuracy')
    #plt.tight_layout()
    #fig.suptitle(f"{model_id} (Loss: {test_loss:.2f}, Acc: {test_acc:.2%})")
    fig.suptitle(model_id)
    if save:
        save_dir = f"{metrics_path}/{model_id}/version_{version}"
        create_folder(save_dir)
        plt.savefig(f'{save_dir}/metrics.png',dpi=200)
    if show:
        plt.show()
    else:
        plt.clf()
        plt.close()

# ---------------------------------------------------------------------------------
        
def save_metrics_from_logger_paths(input_path,output_path):
    metrics = pd.read_csv(input_path) #Only works for train logs currently
    metrics = metrics.drop(['train_loss_step','train_acc_step','train_calibration_error_step'],axis=1)
    metrics = metrics.groupby(metrics['epoch']).first()
    #create_folder(output_path)
    metrics.to_csv(output_path)

# ---------------------------------------------------------------------------------

def save_metrics_from_tensorboard_paths(input_path,output_path):
    accumulator = EventAccumulator(input_path).Reload()
    metrics = pd.DataFrame()
    #chose tags we want - epoch.unique, val_loss val_acc ,train_loss_epoch ,train_acc_epoch 60 
    print(accumulator.scalar_tags)
    epochs = pd.DataFrame(accumulator.Scalars('epoch')).set_index("step").drop(columns="wall_time").drop_duplicates(keep='last').rename(columns={'value': 'epoch'})
    columns = []
    labels = ['val_loss', 'val_acc', 'calibration_error', 'train_loss_epoch', 'train_acc_epoch', 'calibration_error_epoch', 'test_loss', 'test_acc']
    #labels = ['val_loss', 'val_acc', 'train_loss_epoch', 'train_acc_epoch']
    for label in labels:
        columns.append(pd.DataFrame(accumulator.Scalars(label)).set_index("step").drop(columns="wall_time").rename(columns={'value': label}))
    metrics = epochs.join(columns)
    metrics.to_csv(output_path)

# ---------------------------------------------------------------------------------