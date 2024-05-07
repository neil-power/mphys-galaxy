import numpy as np
import pandas as pd
from custom_models.G_ResNet import G_ResNet50
from custom_models.CE_ResNet import CE_Resnet50
from dataset_utils import *
from enum import Enum
import torch
import torch.nn.functional as F
from collections import OrderedDict

def stat_dict_cut(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace("model.","")
        new_state_dict[name] = v
    return new_state_dict

class datasets(Enum):
    FULL_DATASET = 0 #Use all 600,000 galaxies in GZ1 catalog
    CUT_DATASET = 1 #Use cut of 200,000 galaxies, with pre-selected test data and downsampled train data
    BEST_SUBSET = 2 #Select N best S,Z & other galaxies, evenly split
    LOCAL_SUBSET = 3 #Use local cache of 1500 galaxies
    FULL_DESI_DATASET = 4 #Use all 7 million galaxies in DESI catalog, minus those that appear in cut catalog (predict only)
    CUT_TEST_DATASET = 5

class modes(Enum):
    TRAIN = 0 #Train on a dataset
    TEST = 1 #Test an existing saved model on a dataset
    PREDICT = 2 #Use an existing saved model on an unlabelled dataset

DATASET = datasets.CUT_TEST_DATASET #Select which dataset to train on, or if testing/predicting, which dataset the model was trained on
MODE = modes.PREDICT #Select which mode

MULTI = True
MAX_IMG = 1000
BATCHES = [0,1] #[0,1,2,3,4,5,6,7,8,9]

MODEL = "g_resnet50"

device = get_device()

# PATHS = dict(
#     LOCAL_SUBSET_DATA_PATH =  "Data/Subset",
#     LOCAL_SUBSET_CATALOG_PATH =  "Data/gz1_desi_cross_cat_local_subset.csv",
# )

PATHS = dict(
    METRICS_PATH = "/share/nas2/npower/mphys-galaxy/Metrics",
    LOG_PATH = "/share/nas2/npower/mphys-galaxy/Code/lightning_logs",
    FULL_DATA_PATH = '/share/nas2/walml/galaxy_zoo/decals/dr8/jpg',
    LOCAL_SUBSET_DATA_PATH = '/share/nas2/npower/mphys-galaxy/Data/Subset',
    FULL_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat.csv',
    FULL_DESI_CATALOG_PATH =  '/share/nas2/npower/mphys-galaxy/Data/desi_full_cat.parquet',
    CUT_CATALOG_TEST_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_testing.csv',
    CUT_CATALOG_TRAIN_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_train_val_downsample.csv',
    BEST_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_best_subset.csv',
    LOCAL_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_local_subset.csv',
)
# -----------------------------------------------------------------------------

def overlapping(x, y, n=None, beta=0.1): # adapt for 3 classes

    n_z = 100 #what is the significance of 100
    z = np.linspace(0,1,n_z)
    dz = 1./n_z
    
    norm = 1./(beta*np.sqrt(2*np.pi))
    
    n_x = len(x)
    f_x = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_x):
            f_x[i] += norm*np.exp(-0.5*(z[i] - x[j])**2/beta**2)
        f_x[i] /= n_x
    
    
    n_y = len(y)
    f_y = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_y):
            f_y[i] += norm*np.exp(-0.5*(z[i] - y[j])**2/beta**2)
            
        f_y[i] /= n_y

    eta_z = np.zeros(n_z)
    if n is not None:
        n_n = len(n)
        f_n = np.zeros(n_z)
        for i in range(n_z):
            for j in range(n_n):
                f_n[i] += norm*np.exp(-0.5*(z[i] - n[j])**2/beta**2)
                
            f_n[i] /= n_n
        eta_temp = np.minimum(f_x, f_y)
        eta_z = np.minimum(f_n, eta_temp)
    else:
        eta_z = np.minimum(f_x, f_y)
    return np.sum(eta_z)*dz

def fr_rotation_test(model, data, target, idx, device='cpu'):
    #model is the model, data is one image, idx??
    T = 10#50 #number of passes
    rotation_list = range(0, 180, 20)
    
    image_list = []
    outp_list = []
    inpt_list = []
    
    for r in rotation_list:
        
        # make rotated image:
        rotation_matrix = torch.Tensor([[[np.cos(r/360.0*2*np.pi), -np.sin(r/360.0*2*np.pi), 0],
                                         [np.sin(r/360.0*2*np.pi), np.cos(r/360.0*2*np.pi), 0]]]).to(device)
        grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True)
        data_rotate = F.grid_sample(data, grid.to(device), align_corners=True)
        image_list.append(data_rotate.cpu())
        
        # get straight prediction:
        model.eval()
                                         
        # run 100 stochastic forward passes:
        model.enable_dropout_func() #yeah we need to fix this

        output_list, input_list = [], []
        for i in range(T):
            x = model(data_rotate)
            input_list.append(x.unsqueeze(0).cpu())
            output_list.append(torch.unsqueeze(F.softmax(x,dim=1), 0).cpu())
            x=None

        #print("Forward passes for one orientation completed")
                                         
        # append per rotation output into list:
        outp_list.append(np.squeeze(torch.cat(output_list, 0).data.numpy()))
        inpt_list.append(np.squeeze(torch.cat(input_list, 0).data.numpy()))
    
    outp_list = np.array(outp_list)
    inpt_list = np.array(inpt_list)
    rotation_list = np.array(rotation_list)
    
    for i in range(len(rotation_list)):
        x = outp_list[i,:,0]
        y = outp_list[i,:,1]#changed to pick which class overlap to plot
        n = outp_list[i,:,2]
        if MULTI:
            eta = np.zeros((len(rotation_list),3))
            eta_sz =overlapping(x, y)
            eta_zn =overlapping(x, n)
            eta_sn =overlapping(y, n)
            eta[i] =[eta_sz,eta_zn,eta_sn]
        else:
            eta = np.zeros(len(rotation_list))
            eta[i] = overlapping(x, y, n)
    
    if MULTI:
        return np.mean(eta,axis=0), np.std(eta,axis=0)
    else:
        return np.mean(eta), np.std(eta)

datamodule = generate_datamodule(DATASET,MODE,PATHS,datasets,modes,IMG_SIZE=160, NUM_WORKERS=1,BATCH_SIZE=1, MAX_IMAGES=MAX_IMG)
datamodule.prepare_data()
datamodule.setup(stage='predict')

if MODEL == "g_resnet50":
    state_dict = torch.load('Metrics/g_resnet50_cut_dataset_c/version_0/model.pt')
    ResNet = G_ResNet50(num_classes = 2, custom_predict=True, enable_dropout=True)
elif MODEL == "ce_resnet50":
    state_dict = torch.load('Metrics/ce_resnet50_cut_dataset/version_0/model.pt')
    ResNet = CE_Resnet50(enable_dropout = True)

ResNet.load_state_dict(stat_dict_cut(state_dict))
ResNet.to(device)

dataloader_len = len(datamodule.predict_dataset)
total_predict_batches = min(10, dataloader_len)
subset_size = dataloader_len // total_predict_batches

for batch in BATCHES:
    start_idx = batch * subset_size
    end_idx = min((batch + 1) * subset_size, dataloader_len)
    print(f"Loading predict batch {batch} (size {subset_size}) of {total_predict_batches} (size {dataloader_len})")
    for i in range(start_idx,end_idx):

        overlap = pd.DataFrame(columns=['Softmax Prob','Overlap','Err'])
        data1 = torch.Tensor(datamodule.predict_dataset[i]).unsqueeze(0).to(device)

        ResNet.eval()
        p1 = F.softmax(ResNet(data1),dim=1)[0].detach().cpu().numpy()
        av_overlap2, std_overlap2 = fr_rotation_test(ResNet, data=data1, idx = f"img_{i}_{MODEL}", device=device,target=None)
        ce_results =[p1,av_overlap2,std_overlap2]
        overlap.loc[0] = ce_results

        overlap.to_csv(f"rot_err/{'multi/' if MULTI else ''}{MODEL}_overlap_index_{batch}.csv",mode='a', index=False, header=False)