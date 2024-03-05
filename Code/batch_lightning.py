# %% [markdown]
# # Pytorch Lightning for ResNet using galaxy_datasets

# %% [markdown]
# ## Imports

# %%
import os
from enum import Enum
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import lightning as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import albumentations as A
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from ChiralityClassifier import ChiralityClassifier

# %% [markdown]
# ## Options

# %%
class modes(Enum):
    FULL_DATASET = 0 #Use all 600,000 galaxies
    CUT_DATASET = 1 #Use cut of 200,000 galaxies
    BEST_SUBSET = 2 #Select N best S,Z & other galaxies, evenly split
    LOCAL_SUBSET = 3 #Use local cache of 1500 galaxies

IMG_SIZE = 160 # This is the output size of the generated image array
MODE = modes.BEST_SUBSET

#If using best subset, Number of CW, ACW and EL to select
THRESHOLD = 0.8
N_CW = 5000
N_ACW = 5000
N_EL = 5000

FULL_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat.csv'
FULL_DATA_PATH = '/share/nas2/walml/galaxy_zoo/decals/dr8/jpg'
CUT_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_cut.csv'
LOCAL_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/subset_gz1_desi_cross_cat.csv'
LOCAL_SUBSET_DATA_PATH = '/share/nas2/npower/mphys-galaxy/Data/Subset'
SAVE_PATH = "/share/nas2/npower/mphys-galaxy/Models"

torch.set_float32_matmul_precision("medium")

# %% [markdown]
# ## GPU Test

# %%
#Check GPU & Torch is working
print(f"Using pytorch {torch.__version__}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"CPU cores available on device: {os.cpu_count()}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')
print('Using device:', device)

# %% [markdown]
# ## Reading in data

# %% [markdown]
# ### Building catalog

# %%
def get_file_paths(catalog_to_convert,folder_path ):
    brick_ids = catalog_to_convert['dr8_id'].str.split("_",expand=True)[0]
    dr8_ids = catalog_to_convert['dr8_id']
    file_locations = folder_path+'/'+brick_ids+'/'+dr8_ids+'.jpg'
    print(f"Created {file_locations.shape[0]} galaxy filepaths")
    return file_locations

if MODE == modes.FULL_DATASET:
    catalog = pd.read_csv(FULL_CATALOG_PATH)
    catalog['file_loc'] = get_file_paths(catalog,FULL_DATA_PATH)

elif MODE == modes.CUT_DATASET:
    catalog = pd.read_csv(CUT_CATALOG_PATH)
    catalog['file_loc'] = get_file_paths(catalog,FULL_DATA_PATH)

elif MODE == modes.BEST_SUBSET:
    catalog = pd.read_csv(FULL_CATALOG_PATH)
    very_CW_galaxies = catalog[catalog['P_CW']>THRESHOLD]
    very_ACW_galaxies = catalog[catalog['P_ACW']>THRESHOLD]
    very_EL_galaxies = catalog[catalog['P_EL']>THRESHOLD]
    print(f"Very CW: {very_CW_galaxies.shape[0]}, Very ACW: {very_ACW_galaxies.shape[0]}, Very EL: {very_EL_galaxies.shape[0]}")

    galaxy_subset = pd.concat([very_CW_galaxies[0:N_CW],very_ACW_galaxies[0:N_ACW],very_EL_galaxies[0:N_EL]])
    catalog = galaxy_subset.reset_index()
    catalog['file_loc'] = get_file_paths(catalog,FULL_DATA_PATH)

elif MODE == modes.LOCAL_SUBSET:
    catalog = pd.read_csv(LOCAL_SUBSET_CATALOG_PATH)
    catalog['file_loc'] = get_file_paths(catalog,LOCAL_SUBSET_DATA_PATH)

# %% [markdown]
# ### Merging non-S/Z galaxies

# %%
catalog['P_OTHER'] = catalog['P_EL']+catalog['P_EDGE']+catalog['P_DK']+catalog['P_MG']
print(f"Loaded {catalog.shape[0]} galaxy images")

# %% [markdown]
# ## Code to run

# %%
def generate_transforms(resize_after_crop=IMG_SIZE):
    transforms_to_apply = [
        A.ToFloat(), #Converts from 0-255 to 0-1

        A.Resize( #Resizes to 160x160
            height=resize_after_crop,
            width=resize_after_crop,
            interpolation=1,
            always_apply=True
        ),
    ]

    return A.Compose(transforms_to_apply)

datamodule = GalaxyDataModule(
    label_cols=['P_CW','P_ACW','P_OTHER'],
    catalog=catalog,
    train_fraction=0.7,
    val_fraction=0.15,
    test_fraction=0.15,
    custom_albumentation_transform=generate_transforms(),
    batch_size=200,
    num_workers=11,
)

datamodule.prepare_data()
datamodule.setup()

# %%
RUN_TEST = False

# Models:
#resnet18,resnet34,resnet50,resnet101,resnet152,
#jiaresnet50,LeNet,
#G_ResNet18,G_LeNet,

model = ChiralityClassifier(
    num_classes=3, #2 for Jia et al version
    model_version="G_LeNet",
    optimizer="adamw",
    scheduler  ="steplr",
    lr=0.0001,
    weight_decay=0,
    step_size=5,
    gamma=0.85,
    batch_size=60,
)

#stopping_callback = EarlyStopping(monitor="val_loss", mode="min")

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=60,
    devices=1,
    default_root_dir="/share/nas2/npower/mphys-galaxy/Code/",
    profiler="pytorch"
    #callbacks=[stopping_callback]
)

#compiled_model = torch.compile(model, backend="eager")

trainer.fit(model,train_dataloaders=datamodule.train_dataloader(),val_dataloaders=datamodule.val_dataloader() )

if RUN_TEST:
    trainer.test(model,test_dataloader=datamodule.test_dataloader())
    
torch.save(trainer.model.state_dict(), SAVE_PATH + "/lenet_best_subset.pt")


