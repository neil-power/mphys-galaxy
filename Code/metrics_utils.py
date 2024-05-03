## Utilities used for processing, exporting and viewing model metrics

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorboard_reducer.event_loader import EventAccumulator
from dataset_utils import create_folder
from tabulate import tabulate
from numpy.polynomial.polynomial import Polynomial

#          blue      orange     reddish purple  sky blue   bluish green  amber        vermillion  
colours = ["#0072B2","#E69F00","#CC79A7",       "#56B4E9",  "#009E73",  "#D55E00",'black','grey'  ]

# ---------------------------------------------------------------------------------

def save_metrics_from_logger(model_id,log_path,metrics_path,version=0,mode='train',custom_save_id='',save=True):
    metrics = pd.read_csv(f"{log_path}/{model_id}/version_{version}_{mode}{custom_save_id}/metrics.csv")
    if mode == 'train':
        metrics = metrics.drop(['train_loss_step',	'train_acc_step',	'train_calibration_error_step'],axis=1)
    metrics = metrics.groupby(metrics['epoch']).first()
    if save:
        save_dir = f"{metrics_path}/{model_id}/version_{version}"
        create_folder(save_dir)
        metrics.to_csv(f"{save_dir}/{mode}{custom_save_id}_metrics.csv")
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
    
def get_results_runs(model_ids,mode,METRICS_PATH,max_runs=5,clean_titles=True,print_latex=False):
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

        repeat_metrics.loc[model] = {"Loss": f"{(np.average(best_losses) if len(best_losses)>0 else 0):.4f} ± {(np.std(best_losses) if len(best_losses)>0 else 0):.4f}",
                                        "Accuracy": f"{(np.average(best_accs) if len(best_accs)>0 else 0):.2%} ± {(np.std(best_accs) if len(best_accs)>0 else 0):.2%}",
                                        "ECE": f"{(np.average(best_eces) if len(best_eces)>0 else 0):.4f} ± {(np.std(best_eces) if len(best_eces)>0 else 0):.4f}",
                                        "C Viol": f"{(np.average(best_chiralities) if len(best_chiralities)>0 else 0):.4f} ± {(np.std(best_chiralities) if len(best_chiralities)>0 else 0):.4f}"}
    if print_latex:
        print(tabulate(repeat_metrics,headers='keys',tablefmt='latex'))
    if clean_titles:
        repeat_metrics.index = repeat_metrics.index.str.replace('_cut_dataset','')
    return repeat_metrics

# ---------------------------------------------------------------------------------

def get_predict_results_runs(model_ids,max_runs,METRICS_PATH,dataset_name="full_desi_dataset",clean_titles=True,print_latex=False,max_batch=10,errors=True):
    repeat_metrics = pd.DataFrame(columns=["ACW","CW","Other","S/Z Ratio","C Viol"],index=model_ids)
    repeat_metrics.columns.name="Model"
    for model in model_ids:
        total = 0
        acws = []
        cws = []
        others = []
        c_viols = []
        ratios = []
        for run in range(max_runs):
            num = 0
            num_cw = 0
            num_acw = 0
            num_other = 0
            for batch in range(max_batch):
                try:
                    predictions = pd.read_csv(f"{METRICS_PATH}/{model}/version_{run}/{dataset_name}_{batch}_predictions.csv",header=None,names=['CW','ACW','Other']).astype('float')#, on_bad_lines = 'skip'
                    num += predictions.shape[0]
                    num_cw += np.count_nonzero(predictions['CW']>0.5)
                    num_acw += np.count_nonzero(predictions['ACW']>0.5)
                    num_other += predictions.shape[0] - np.count_nonzero(predictions['ACW']>0.5) - np.count_nonzero(predictions['CW']>0.5)
                except:
                    if errors:
                        print(f"Error with {model}, run {run}, batch {batch}")
            acws.append(num_acw)
            cws.append(num_cw)
            others.append(num_other)
            c_viol = (num_acw-num_cw)/np.sqrt(num_cw+num_acw)
            c_viols.append(c_viol)
            if num_acw !=0 and num_cw !=0:
                ratios.append(num_acw/num_cw)
            else:
                ratios.append(0)
            total += num
        total /=max_runs
        repeat_metrics.loc[model] = {"ACW": f"{np.average(acws):.0f} ({np.average(acws)/total:.1%}) ± {np.std(acws):.0f}",
                                        "CW": f"{np.average(cws):.0f} ({np.average(cws)/total:.1%}) ± {np.std(cws):.0f}",
                                        "Other": f"{np.average(others):.0f} ({np.average(others)/total:.1%}) ± {np.std(others):.0f}",
                                        "S/Z Ratio": f"{np.average(ratios):3.2f} ± {np.std(ratios):3.2f}",
                                        "C Viol": f"{np.average(c_viols):3.2f} ± {np.std(c_viols):3.2f}"}
    if print_latex:
        print(tabulate(repeat_metrics,headers='keys',tablefmt='latex'))
    if clean_titles:
        repeat_metrics.index = repeat_metrics.index.str.replace('_cut_dataset','')
    return repeat_metrics

# ---------------------------------------------------------------------------------

def chirality_violation(labels):
    # CW,ACW, OTHER
    n_z = np.count_nonzero(labels['CW']>0.5)
    n_s = np.count_nonzero(labels['ACW']>0.5)
    return (n_s-n_z)/np.sqrt(n_z+n_s)

# ---------------------------------------------------------------------------------

def count_spirals(labels):
    # CW,ACW, OTHER
    n_z = np.count_nonzero(labels['CW']>0.5)
    n_s = np.count_nonzero(labels['ACW']>0.5)
    return n_z+n_s

# ---------------------------------------------------------------------------------


def get_predict_results_runs_cviol(model_ids,c_viols_list,METRICS_PATH,max_runs=5,dataset_name="cut_test_dataset"):
    repeat_metrics = pd.DataFrame(columns=["C Viol", "C Viol Err","N Spirals","N Spirals Err"],index=model_ids)
    repeat_metrics.columns.name="Model"
    for model in model_ids:
        c_viols = []
        c_viols_err = []
        num_spirals = []
        num_spirals_err = []
        for c_viol in c_viols_list:
            predicted_c_viols = []
            predicted_num_spirals = []
            for run in range(max_runs):
                predictions = pd.read_csv(f"{METRICS_PATH}/{model}/version_{run}/{dataset_name}_CVIOL_{c_viol}_predictions.csv",header=None,names=['CW','ACW','Other'], on_bad_lines = 'skip').astype('float')
                predicted_c_viols.append(chirality_violation(predictions))
                predicted_num_spirals.append(count_spirals(predictions))
            c_viols.append(np.average(predicted_c_viols))
            c_viols_err.append(np.std(predicted_c_viols))
            num_spirals.append(np.average(predicted_num_spirals))
            num_spirals_err.append(np.std(predicted_num_spirals))
        
        repeat_metrics.loc[model] = {"C Viol": c_viols,
                                        "C Viol Err": c_viols_err,
                                        "N Spirals": num_spirals,
                                        "N Spirals Err": num_spirals_err}
    return repeat_metrics

# ---------------------------------------------------------------------------------

def plot_cviols(repeat_metrics,model_ids,c_viols_list,title=True,dims=(9,11)):
    
    if len(model_ids) == 1:
        if dims == (9,11):
            dims = (6,4)
        fig = plt.figure(figsize=dims)
        xsize = 1
        ysize=1
    else:
        fig = plt.figure(figsize=dims)
        xsize = int(len(model_ids)/2)+1
        ysize=2
        
    for i,model in enumerate(model_ids):
        ax = fig.add_subplot(xsize,ysize,i+1)
        ax.set_ylabel('Predicted C Viol',fontsize=14)
        ax.set_xlabel('Actual C Viol',fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=14)
        c_viols = repeat_metrics["C Viol"].iloc[i]
        c_viols_err = repeat_metrics["C Viol Err"].iloc[i]
        label = model.replace('_cut_dataset','')
        label = label.replace('_repeat','')
        ax.errorbar(c_viols_list,c_viols,yerr=c_viols_err,fmt="x",linewidth=1.7,capsize=5) #label=label
        ax.plot(c_viols_list,c_viols_list,linewidth=1.7)#,label="Ideal")
        fit = Polynomial.fit(c_viols_list,c_viols,deg=1,w=c_viols_err)
        ax.plot(*fit.linspace(10),label=f"y = {fit.convert().coef[1]:3.2f} x + {fit.convert().coef[0]:3.2f}")
        ax.grid()
        if title:
            ax.set_title(label)
        ax.set_xticks(c_viols_list)
        ax.legend(fontsize=14)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------------

def plot_spiral_nums(repeat_metrics,model_ids,c_viols_list):
    fig = plt.figure(figsize=(9,11))

    for i,model in enumerate(model_ids):
        ax = fig.add_subplot(int(len(model_ids)/2)+1,2,i+1)
        ax.set_ylabel('S & Z Predicted')
        ax.set_xlabel('Actual C Viol')
        num_actual = 2333
        num_spirals = np.array(repeat_metrics["N Spirals"].iloc[i])/num_actual
        num_spirals_err = np.array(repeat_metrics["N Spirals Err"].iloc[i])/num_actual
        label = model.replace('_cut_dataset','')
        label = label.replace('_repeat','')
        ax.errorbar(c_viols_list,num_spirals,yerr=num_spirals_err,fmt="x",linewidth=1.7,capsize=5) #label=label
        ax.grid()
        ax.set_title(label)
        ax.set_xticks(c_viols_list)
        ax.set_yticks(np.arange(0.2,1.2,0.2))
    plt.tight_layout()
    plt.show()

 # ---------------------------------------------------------------------------------   
    
def plot_combined_metrics(model_ids,metric_to_use,metric_label,METRICS_PATH,max_runs=5,show_err=True,l_loc='lower right'):
    fig = plt.figure(figsize=(14,6))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Epoch',fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=10)
    ax1.set_ylabel(metric_label,fontsize=14)

    for i,model in enumerate(model_ids):
        model_metrics = []
        for run in range(max_runs):
            run_metrics = get_metrics_from_csv(model,METRICS_PATH,version=run,mode='train')
            run_metric_to_use = run_metrics[metric_to_use]
            model_metrics.append(run_metric_to_use)
        
        epoch = run_metrics.index #Get from last run
        avg_metrics = np.average(np.array(model_metrics),axis=0)
        std_metrics = np.std(np.array(model_metrics),axis=0)
        label = model.replace('_cut_dataset','').replace('g','G').replace('ce','CE').replace('lenet','LeNet').replace('resnet','ResNet')
        ax1.plot(epoch,avg_metrics,linestyle='-',c=colours[i],label=label,linewidth=2)
        if show_err:
            ax1.plot(epoch,avg_metrics-std_metrics,linestyle='--',c=colours[i],linewidth=.5)
            ax1.plot(epoch,avg_metrics+std_metrics,linestyle='--',c=colours[i],linewidth=.5)

    ax1.legend(loc=l_loc,ncol=int(len(model_ids)/2),fontsize=14)
    ax1.grid()
    plt.show()

 # ---------------------------------------------------------------------------------   
    
def plot_combined_metrics_run(model_ids,metric,metric_label,METRICS_PATH,version=0):
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(111)
    ax1.set_xlabel('Epoch',fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=14)
    #ax.tick_params(axis='both', which='minor', labelsize=10)
    ax1.set_ylabel(metric_label,fontsize=14)

    for i,model in enumerate(model_ids):
        metrics = get_metrics_from_csv(model,METRICS_PATH,version=version,mode='train')
        m = metrics[metric]
        epoch = metrics.index
        label = model.replace('_cut_dataset','').replace('g','G').replace('ce','CE').replace('lenet','LeNet').replace('resnet','ResNet')
        ax1.plot(epoch,m,linestyle='-',c=colours[i],label=label,linewidth=1.7)
    ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5),fontsize=14)
    ax1.grid()
    plt.show()