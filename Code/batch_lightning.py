# %% [markdown]
# # Pytorch Lightning for ResNet using galaxy_datasets

# %% [markdown]
# ## Imports

# %%
import gc
from enum import Enum
import torch
import lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger,CSVLogger
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from ChiralityClassifier import ChiralityClassifier
from dataset_utils import *
from metrics_utils import *

# %% [markdown]
# ## Options

# %%
class datasets(Enum):
    FULL_DATASET = 0 #Use all 600,000 galaxies in GZ1 catalog
    CUT_DATASET = 1 #Use cut of 200,000 galaxies, with pre-selected test data and downsampled train data
    BEST_SUBSET = 2 #Select N best S,Z & other galaxies, evenly split
    LOCAL_SUBSET = 3 #Use local cache of 1500 galaxies
    FULL_DESI_DATASET = 4 #Use all 7 million galaxies in DESI catalog, minus those that appear in cut catalog (predict only)

class modes(Enum):
    TRAIN = 0 #Train on a dataset
    TEST = 1 #Test an existing saved model on a dataset
    PREDICT = 2 #Use an existing saved model on an unlabelled dataset

DATASET = datasets.CUT_DATASET #Select which dataset to run
MODE = modes.TEST #Select which mode

# Models:
#resnet18,resnet34,resnet50,resnet101,resnet152,
#jiaresnet50,lenet,g_resnet18,g_lenet,
MODEL_NAME = "g_lenet"
CUSTOM_ID = "repeat"

USE_TENSORBOARD = False #Log to tensorboard as well as csv logger
SAVE_MODEL = True #Save model weights to .pt file
REPEAT_RUNS = 5 #Set to 1 for 1 run
IMG_SIZE = 160 #This is the output size of the generated image array
BATCH_SIZE = 100 #Number of images per batch
NUM_WORKERS = 11 #Number of workers in dataloader (no of CPU cores - 1)

PATHS = dict(
    METRICS_PATH = "/share/nas2/npower/mphys-galaxy/Metrics",
    LOG_PATH = "/share/nas2/npower/mphys-galaxy/Code/lightning_logs",
    FULL_DATA_PATH = '/share/nas2/walml/galaxy_zoo/decals/dr8/jpg',
    LOCAL_SUBSET_DATA_PATH = '/share/nas2/npower/mphys-galaxy/Data/Subset',
    FULL_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat.csv',
    CUT_CATALOG_TEST_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_testing.csv',
    CUT_CATALOG_TRAIN_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_train_val_downsample.csv',
    BEST_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_best_subset.csv',
    LOCAL_SUBSET_CATALOG_PATH = '/share/nas2/npower/mphys-galaxy/Data/gz1_desi_cross_cat_local_subset.csv',
)

torch.set_float32_matmul_precision("medium")
if len(CUSTOM_ID) == 0:
    MODEL_ID = f"{MODEL_NAME}_{DATASET.name.lower()}"
else:
     MODEL_ID = f"{MODEL_NAME}_{DATASET.name.lower()}_{CUSTOM_ID}"

if MODE != modes.TRAIN:
    USE_TENSORBOARD = False #Don"t log to tensorboard if not training
    SAVE_MODEL = False #Don"t save weights if testing or predicting model
# %% [markdown]
# ## GPU Test

# %% 
device = get_device()

# %% [markdown]
# ## Reading in data

# %% [markdown]
# ### Building catalog
datamodule = generate_datamodule(DATASET,MODE,PATHS,datasets,modes,IMG_SIZE,BATCH_SIZE,NUM_WORKERS)

# %% [markdown]
# ## Code to run

# %%
datamodule.prepare_data()
if MODE == modes.PREDICT:
    datamodule.setup(stage='predict')
else:
    datamodule.setup()

# %%
for run in range(0,REPEAT_RUNS):
    
    save_dir = f"{PATHS['METRICS_PATH']}/{MODEL_ID}/version_{run}"
    MODEL_PATH = f"{save_dir}/model.pt"
    create_folder(save_dir)

    model = ChiralityClassifier(
        num_classes=(2 if (MODEL_NAME=="jiaresnet50") else 3), #2 for Jia et al version
        model_version=MODEL_NAME,
        optimizer="adamw",
        scheduler  ="steplr",
        lr=0.0001,
        weight_decay=0,
        step_size=5,
        gamma=0.85,
        weights=(MODEL_PATH if MODE != modes.TRAIN else None),
        graph_save_path=(f"{save_dir}/val_matrix.png" if MODE == modes.TRAIN else f"{save_dir}/{MODE.name.lower()}_matrix.png")
    )

    tb_logger = TensorBoardLogger(PATHS["LOG_PATH"], name=MODEL_ID,version=f"version_{run}_{MODE.name.lower()}")
    csv_logger = CSVLogger(PATHS["LOG_PATH"],name=MODEL_ID,version=f"version_{run}_{MODE.name.lower()}")
    trainer = pl.Trainer(
        accelerator=("gpu" if device.type=="cuda" else "cpu"),
        max_epochs=60,
        devices=1,
        logger=([tb_logger,csv_logger] if USE_TENSORBOARD else csv_logger),
        default_root_dir=f"{PATHS['LOG_PATH']}/{MODEL_ID}",
        enable_checkpointing=False,
        #profiler="pytorch"
        #callbacks=EarlyStopping(monitor="val_loss", mode="min")
    )

    #compiled_model = torch.compile(model, backend="eager")
    
    if MODE==modes.TRAIN:
        trainer.fit(model,train_dataloaders=datamodule.train_dataloader(),val_dataloaders=datamodule.val_dataloader())
        trainer.test(model,dataloaders=datamodule.val_dataloader())

        if SAVE_MODEL:
            torch.save(trainer.model.state_dict(), MODEL_PATH)
        
    elif MODE==modes.TEST:
        trainer.test(model,dataloaders=datamodule.test_dataloader())
           
    elif MODE==modes.PREDICT:
        trainer.predict(model,dataloaders=datamodule.predict_dataloader())   

    #Save cleaned up logs file to Metrics folder & save graph
    save_metrics_from_logger(MODEL_ID,PATHS["LOG_PATH"],PATHS['METRICS_PATH'],version=run,mode=MODE.name.lower(),save=True)  
    if MODE==modes.TRAIN:
        plot_train_metrics(MODEL_ID,PATHS['METRICS_PATH'],version=run,show=False,save=True)     

# %%
#Dereference all objects, clear cuda cache and run garbage collection
datamodule=None
model=None
trainer=None
with torch.no_grad():
    torch.cuda.empty_cache()
gc.collect()