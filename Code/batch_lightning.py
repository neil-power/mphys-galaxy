# %% [markdown]
# # Pytorch Lightning for ResNet using galaxy_datasets

# %% [markdown]
# ## Imports

# %%
import os
from enum import Enum
import pandas as pd
import torch
import lightning as pl
from torch.utils.data import random_split
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import albumentations as A
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule
from ChiralityClassifier import ChiralityClassifier

# %% [markdown]
# ## Options

# %%
class modes(Enum):
    FULL_DATASET = 0 #Use all 600,000 galaxies
    CUT_DATASET = 1 #Use cut of 200,000 galaxies, with pre-selected test data and downsampled train data
    BEST_SUBSET = 2 #Select N best S,Z & other galaxies, evenly split
    LOCAL_SUBSET = 3 #Use local cache of 1500 galaxies

IMG_SIZE = 160 # This is the output size of the generated image array
MODE = modes.CUT_DATASET
RUN_TEST = False #Run on testing dataset & save metrics
# Models:
#resnet18,resnet34,resnet50,resnet101,resnet152,
#jiaresnet50,LeNet,
#G_ResNet18,G_LeNet,
MODEL_NAME = 'G_LeNet'
CUSTOM_ID = ''

#For .py files
MODEL_SAVE_PATH = "/share/nas2/npower/mphys-galaxy/Models"
GRAPH_SAVE_PATH = "/share/nas2/npower/mphys-galaxy/Graphs"
LOG_PATH = "/share/nas2/npower/mphys-galaxy/Code/"
FULL_DATA_PATH = '/share/nas2/walml/galaxy_zoo/decals/dr8/jpg'
LOCAL_SUBSET_DATA_PATH = '/share/nas2/npower/mphys-galaxy/Data/Subset'

FULL_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat.csv'
CUT_CATALOG_TEST_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_testing.csv'
CUT_CATALOG_TRAIN_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_train_val_downsample.csv'
BEST_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_best_subset.csv'
LOCAL_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_local_subset.csv'

torch.set_float32_matmul_precision("medium")
MODEL_ID = f"{MODEL_NAME}_{MODE.name.lower()}_{CUSTOM_ID}"

# %% [markdown]
# ## GPU Test

# %%
print(f"Using pytorch {torch.__version__}. CPU cores available on device: {os.cpu_count()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print(f'Allocated Memory: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB')
    print(f'Cached Memory: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB')
print('Using device:', device)

# %% [markdown]
# ## Reading in data

# %% [markdown]
# ### Building catalog

# %%
def get_file_paths(catalog_to_convert,folder_path):
    brick_ids = catalog_to_convert['dr8_id'].str.split("_",expand=True)[0]
    dr8_ids = catalog_to_convert['dr8_id']
    file_locations = folder_path+'/'+brick_ids+'/'+dr8_ids+'.jpg'
    print(f"Created {file_locations.shape[0]} galaxy filepaths")
    return file_locations

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

# %% [markdown]
# ### Merging non-S/Z galaxies

# %%
if MODE == modes.CUT_DATASET:
    train_val_catalog = pd.read_csv(CUT_CATALOG_TRAIN_PATH)
    train_val_catalog['file_loc'] = get_file_paths(train_val_catalog,FULL_DATA_PATH)
    generator1 = torch.Generator()
    train_catalog, val_catalog = random_split(train_val_catalog, [0.20,0.80], generator=generator1)
    train_catalog = train_catalog.dataset.iloc[train_catalog.indices]
    val_catalog = val_catalog.dataset.iloc[val_catalog.indices]   
    test_catalog = pd.read_csv(CUT_CATALOG_TEST_PATH)
    test_catalog['file_loc'] = get_file_paths(test_catalog,FULL_DATA_PATH)

    datamodule = GalaxyDataModule(
        label_cols=['P_CW','P_ACW','P_OTHER'],
        train_catalog=train_catalog, val_catalog=train_catalog, test_catalog=test_catalog,
        custom_albumentation_transform=generate_transforms(),
        batch_size=100,
        num_workers=11,
    )
    
else:
    if MODE == modes.FULL_DATASET:
        catalog = pd.read_csv(FULL_CATALOG_PATH)
        catalog['file_loc'] = get_file_paths(catalog,FULL_DATA_PATH)

    elif MODE == modes.BEST_SUBSET:
        catalog = pd.read_csv(BEST_SUBSET_CATALOG_PATH)
        catalog['file_loc'] = get_file_paths(catalog,FULL_DATA_PATH)

    elif MODE == modes.LOCAL_SUBSET:
        catalog = pd.read_csv(LOCAL_SUBSET_CATALOG_PATH)
        catalog['file_loc'] = get_file_paths(catalog,LOCAL_SUBSET_DATA_PATH)

    datamodule = GalaxyDataModule(
        label_cols=['P_CW','P_ACW','P_OTHER'],
        catalog=catalog,
        train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
        custom_albumentation_transform=generate_transforms(),
        batch_size=100,
        num_workers=11,
    )

# %% [markdown]
# ## Code to run

# %%
datamodule.prepare_data()
datamodule.setup()

# %%
RUN_TEST = False

# Models:
#resnet18,resnet34,resnet50,resnet101,resnet152,
#jiaresnet50,LeNet,
#G_ResNet18,G_LeNet,

model = ChiralityClassifier(
    num_classes=(2 if (MODEL_NAME=="jiaresnet50") else 3), #2 for Jia et al version
    model_version=MODEL_NAME,
    optimizer="adamw",
    scheduler  ="steplr",
    lr=0.0001,
    weight_decay=0,
    step_size=5,
    gamma=0.85,
    batch_size=60,
    weights=None,
    model_save_path=f"{MODEL_SAVE_PATH}/{MODEL_ID}.pt",
    graph_save_path=f"{GRAPH_SAVE_PATH}/{MODEL_ID}_matrix.png"
)

#stopping_callback = EarlyStopping(monitor="val_loss", mode="min")

trainer = pl.Trainer(
    accelerator=("gpu" if device.type=="cuda" else "cpu"),
    max_epochs=60,
    devices=1,
    default_root_dir=LOG_PATH,
    profiler="pytorch"
    #callbacks=[stopping_callback]
)

#compiled_model = torch.compile(model, backend="eager")
trainer.fit(model,train_dataloaders=datamodule.train_dataloader(),val_dataloaders=datamodule.val_dataloader() )

if RUN_TEST:
    trainer.test(model,dataloaders=datamodule.test_dataloader())
else:
    trainer.test(model,dataloaders=datamodule.val_dataloader())
    
torch.save(trainer.model.state_dict(), model.model_save_path)


