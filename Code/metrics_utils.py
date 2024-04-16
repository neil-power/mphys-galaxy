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
    
def get_results_runs(model_ids,mode,METRICS_PATH,max_runs=5):
    repeat_metrics = pd.DataFrame(columns=["Loss","Accuracy","ECE","C Viol"],index=model_ids)
    repeat_metrics.columns.name="Model"
    for model in model_ids:
        best_losses = []
        best_accs = []
        best_eces = []
        best_chiralities = []
        for run in range(max_runs):
            try:
                if mode =='val':
                    metrics = get_metrics_from_csv(model,METRICS_PATH,version=run,mode='train')
                    best_loss_epoch = metrics['val_loss'].argmin()
                    best_losses.append(metrics['val_loss'][best_loss_epoch])
                    best_accs.append(metrics['val_acc'][best_loss_epoch])
                    best_eces.append(metrics['val_calibration_error'][best_loss_epoch])
                    best_chiralities.append((metrics[f'{mode}_chirality_violation'][best_loss_epoch]))
                elif mode =='test':
                    metrics = get_metrics_from_csv(model,METRICS_PATH,version=run,mode=mode)
                    best_losses.append(metrics['test_loss'])
                    best_accs.append(metrics['test_acc'])
                    best_eces.append(metrics['test_calibration_error'])
                    best_chiralities.append((metrics['test_chirality_violation']))
            except:
                print(f"Error with {model}, run {run}")

        nans_removed = np.count_nonzero(np.isnan(np.concatenate((best_losses, best_accs, best_eces, best_chiralities))))
        if nans_removed > 0:
            print(f"{model}: Removed {nans_removed} NaNs")
        best_losses = np.array(best_losses)[~np.isnan(best_losses)]
        best_accs = np.array(best_accs)[~np.isnan(best_accs)]
        best_eces = np.array(best_eces)[~np.isnan(best_eces)]
        best_chiralities = np.array(best_chiralities)[~np.isnan(best_chiralities)]

        repeat_metrics.loc[model] = {"Loss": f"{np.average(best_losses):.4f} ± {np.std(best_losses):.4f}",
                                        "Accuracy": f"{np.average(best_accs):.2%} ± {np.std(best_accs):.2%}",
                                        "ECE": f"{np.average(best_eces):.4f} ± {np.std(best_eces):.4f}",
                                        "C Viol": f"{np.average(best_chiralities):.4f} ± {np.std(best_chiralities):.4f}"}
    #print(tabulate(repeat_metrics,headers='keys',tablefmt='github'))
    return repeat_metrics

# ---------------------------------------------------------------------------------

def get_predict_results_runs(model_ids,max_runs,METRICS_PATH,dataset_name="full_desi_dataset"):
    repeat_metrics = pd.DataFrame(columns=["ACW","CW","Other","C Viol"],index=model_ids)
    repeat_metrics.columns.name="Model"
    for model in model_ids:
        acws = []
        cws = []
        others = []
        c_viols = []
        for run in range(max_runs):
            try:
                predictions = pd.read_csv(f"{METRICS_PATH}/{model}/version_{run}/{dataset_name}_predictions.csv",header=None,names=['CW','ACW','Other'], on_bad_lines = 'skip').astype('float')
                argmax_predictions = predictions.idxmax(axis=1)
                num_acw = argmax_predictions[argmax_predictions=='ACW'].shape[0]
                num_cw = argmax_predictions[argmax_predictions=='CW'].shape[0]
                num_other = argmax_predictions[argmax_predictions=='Other'].shape[0]
                #num = argmax_predictions.shape[0]
                acws.append(num_acw)
                cws.append(num_cw)
                others.append(num_other)
                c_viols.append(chirality_violation(predictions))
            except:
                print(f"Error with {model}, run {run}")
        
        
        repeat_metrics.loc[model] = {"ACW": f"{np.average(acws):.0f} ({np.average(acws)/1e6:.1%}) ± {np.std(acws):.0f}",
                                        "CW": f"{np.average(cws):.0f} ({np.average(cws)/1e6:.1%}) ± {np.std(cws):.0f}",
                                        "Other": f"{np.average(others):.0f} ({np.average(others)/1e6:.1%}) ± {np.std(others):.0f}",
                                        "C Viol": f"{np.average(c_viols):3.2f} ± {np.std(c_viols):3.2f}"}
    #print(tabulate(repeat_metrics,headers='keys',tablefmt='github'))
    return repeat_metrics

# ---------------------------------------------------------------------------------

def chirality_violation(labels):
    # CW,ACW, OTHER
    n_z = np.count_nonzero(labels['CW']>0.5)
    n_s = np.count_nonzero(labels['ACW']>0.5)
    return (n_s-n_z)/np.sqrt(n_z+n_s)

# ---------------------------------------------------------------------------------