import numpy as np
import pylab as pl
import pandas as pd
from custom_models.G_ResNet import G_ResNet50
from custom_models.CE_ResNet import CE_Resnet50
from dataset_utils import *
from enum import Enum

import torch
import torch.nn.functional as F
from collections import OrderedDict

from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)

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

DATASET = datasets.LOCAL_SUBSET #Select which dataset to train on, or if testing/predicting, which dataset the model was trained on
MODE = modes.PREDICT #Select which mode

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

def build_mask(s, margin=2, dtype=torch.float32):
    mask = torch.zeros(1, 1, s, s, dtype=dtype)
    c = (s-1) / 2
    t = (c - margin/100.*c)**2
    sig = 2.
    for x in range(s):
        for y in range(s):
            r = (x - c) ** 2 + (y - c) ** 2
            if r > t:
                mask[..., x, y] = math.exp((t - r)/sig**2)
            else:
                mask[..., x, y] = 1.
    return mask
    
def positionimage(x, y, ax, ar, zoom=0.5):
    """Place image from file `fname` into axes `ax` at position `x,y`."""
    
    imsize = ar.shape[0]
    if imsize==151: zoom=0.24
    if imsize==51: zoom = 0.75
    im = OffsetImage(ar, zoom=zoom)
    im.image.axes = ax
    
    ab = AnnotationBbox(im, (x,y), xycoords='data')
    ax.add_artist(ab)
    
    return

# -----------------------------------------------------------------------------

def make_linemarker(x,y,dx,col,ax):
    
    xs = [x-0.5*dx,x+0.5*dx]
    for i in range(0,y.shape[0]):
        ys = [y[i],y[i]]
        ax.plot(xs,ys,marker=",",c=col,alpha=0.1,lw=5)
    
    return

def overlapping(x, y, n, beta=0.1): # adapt for 3 classes

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


    n_n = len(n) #third class like this?
    f_n = np.zeros(n_z)
    for i in range(n_z):
        for j in range(n_n):
            f_n[i] += norm*np.exp(-0.5*(z[i] - n[j])**2/beta**2)
            
        f_n[i] /= n_n
    
    
    eta_z = np.zeros(n_z)
    eta_temp = np.minimum(f_x, f_y)
    eta_z = np.minimum(f_n, eta_temp)

    return np.sum(eta_z)*dz

def fr_rotation_test(model, data, target, idx, device='cpu', PLOT=False):
    #model is the model, data is one image, idx??
    T = 50#50 #number of passes
    rotation_list = range(0, 180, 20)
    #print("True classification: ",target[0].item())
    
    image_list = []
    outp_list = []
    inpt_list = []
    
    for r in rotation_list:
        
        # make rotated image:
        rotation_matrix = torch.Tensor([[[np.cos(r/360.0*2*np.pi), -np.sin(r/360.0*2*np.pi), 0],
                                         [np.sin(r/360.0*2*np.pi), np.cos(r/360.0*2*np.pi), 0]]]).to(device)
        grid = F.affine_grid(rotation_matrix, data.size(), align_corners=True) #data.size()
        data_rotate = F.grid_sample(data, grid, align_corners=True)
        image_list.append(data_rotate)
        
        # get straight prediction:
        model.eval()
        #x = model(data_rotate)
        #p = F.softmax(x,dim=1)
                                         
        # run 100 stochastic forward passes:
        model.enable_dropout_func() #yeah we need to fix this

        output_list, input_list = [], []
        for i in range(T):
            #print("hi"+str(i))
            x = model(data_rotate)
            input_list.append(x.unsqueeze(0).cpu())
            output_list.append(torch.unsqueeze(F.softmax(x,dim=1), 0).cpu())
            x=None

        print("Forward passes for one orientation completed")
                                         
        # append per rotation output into list:
        outp_list.append(np.squeeze(torch.cat(output_list, 0).data.numpy()))
        inpt_list.append(np.squeeze(torch.cat(input_list, 0).data.numpy()))

    preds = np.array([0,1,2])#
    classes = np.array(["Z-wise","S-wise", "None"])#
    
    outp_list = np.array(outp_list)
    inpt_list = np.array(inpt_list)
    rotation_list = np.array(rotation_list)

    colours=["b","r","g"]#
    fig2, (a2, a3) = pl.subplots(2, 1, gridspec_kw={'height_ratios': [8,1]})

    eta = np.zeros(len(rotation_list))
    for i in range(len(rotation_list)):
        x = outp_list[i,:,0]
        y = outp_list[i,:,1]#changed to pick which class overlap to plot
        n = outp_list[i,:,2]
        eta[i] = overlapping(x, y, n)

    if PLOT:
        #a0.set_title("Input")
        if np.mean(eta)>=0.01:
            a2.set_title(r"$\langle \eta \rangle = $ {:.2f}".format(np.mean(eta)))
        else:
            a2.set_title(r"$\langle \eta \rangle < 0.01$")

        dx = 0.8*(rotation_list[1]-rotation_list[0])
        for pred in preds:
            col = colours[pred]
            #a0.plot(rotation_list[0],inpt_list[0,0,pred],marker=",",c=col,label=str(pred))
            a2.plot(rotation_list[0],outp_list[0,0,pred],marker=",",c=col,label=classes[pred])
            for i in range(rotation_list.shape[0]):
                #offset = np.random.rand(*outp_list[0,:,pred].shape)-0.5
            #    make_linemarker(rotation_list[i],inpt_list[i,:,pred],dx,col,a0)
                make_linemarker(rotation_list[i],outp_list[i,:,pred],dx,col,a2)
            
        #a2.plot(rotation_list, eta)
        
        #a0.legend()
        a2.legend(loc='center right')
        #a0.axis([0,180,0,1])
        #a0.set_xlabel("Rotation [deg]")
        a2.set_xlabel("Rotation [deg]")
        #a1.axis([0,180,0,1])
        a3.axis([0,180,0,1])
        #a1.axis('off')
        a3.axis('off')
        
        imsize = data.size()[2]
        mask = build_mask(imsize, margin=1)
                
        for i in range(len(rotation_list)):
            inc = 0.5*(180./len(rotation_list))
            #positionimage(rotation_list[i]+inc, 0., a1, image_list[i][0, 0, :, :].data.numpy(), zoom=0.32)
            positionimage(rotation_list[i]+inc, 0., a3, mask[0,0,:,:]*image_list[i][0, 0, :, :].data.cpu().numpy(), zoom=0.23)#make it smaller
            
        
        #fig1.tight_layout()
        fig2.tight_layout()

        #fig1.subplots_adjust(bottom=0.15)
        fig2.subplots_adjust(bottom=0.15)

        #pl.show()
        fig2.savefig("../rot_err/"+str(idx)+".png")
    
        pl.close()
        print("Plot generated")
    
    return np.mean(eta), np.std(eta)

G_resnet= G_ResNet50(num_classes = 2, custom_predict=True, enable_dropout=True)
CE_resnet= CE_Resnet50(enable_dropout = True)

state_dict_G = torch.load('../Metrics/g_resnet50_cut_dataset_c/version_0/model.pt')
state_dict_CE = torch.load('../Metrics/CE_resnet50_cut_dataset/version_0/model.pt')

G_resnet.load_state_dict(stat_dict_cut(state_dict_G))
CE_resnet.load_state_dict(stat_dict_cut(state_dict_CE))
print("Models loaded")

datamodule = generate_datamodule(DATASET,MODE,PATHS,datasets,modes,IMG_SIZE=160, NUM_WORKERS=1,BATCH_SIZE=1, MAX_IMAGES=10)
datamodule.prepare_data()
datamodule.setup(stage='predict')
print("Dataset primed")

overlap_results = pd.DataFrame(columns=['Steerable Overlap','Steerable Err','CE Overlap','CE Err'])
i=0
for data1 in datamodule.predict_dataloader():
    print("Begining assesment on image "+str(i))
    av_overlap2, std_overlap2 = fr_rotation_test(CE_resnet, data=data1, idx ="resnet_CE_"+str(i), target=None, PLOT=True)
    print("Resnet run completed")
    av_overlap1, std_overlap1 = fr_rotation_test(G_resnet, data=data1, idx ="resnet_G_"+ str(i), target=None, PLOT=True)
    print("G_Resnet run completed")
    results =[av_overlap1,std_overlap1,av_overlap2,std_overlap2]
    overlap_results.loc[i] = results
    i+=1
overlap_results.to_csv("Overlap Index")
print("Results writted to CSV")