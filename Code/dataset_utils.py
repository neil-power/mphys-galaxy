## Utilities used for data processing and data loading

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import random_split
import albumentations as A
from galaxy_datasets.pytorch.galaxy_datamodule import GalaxyDataModule

# ---------------------------------------------------------------------------------

def get_file_paths(catalog_to_convert,folder_path):
    brick_ids = catalog_to_convert["dr8_id"].str.split("_",expand=True)[0]
    dr8_ids = catalog_to_convert["dr8_id"]
    file_locations = folder_path+"/"+brick_ids+"/"+dr8_ids+".jpg"
    print(f"Created {file_locations.shape[0]} galaxy filepaths")
    return file_locations

# ---------------------------------------------------------------------------------

def create_folder(save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

# ---------------------------------------------------------------------------------

def save_metrics_from_logger(model_id,version,mode,log_path,metrics_path):
    metrics = pd.read_csv(f"{log_path}/{model_id}/version_{version}/metrics.csv")
    metrics = metrics.drop(['train_loss_step',	'train_acc_step',	'train_calibration_error_step'],axis=1)
    metrics = metrics.groupby(metrics['epoch']).first()
    save_dir = f"{metrics_path}/{model_id}/version_{version}"
    create_folder(save_dir)
    metrics.to_csv(f"{save_dir}/{mode}_metrics.csv")

# ---------------------------------------------------------------------------------

def generate_transforms(resize_after_crop=160):
    transforms_to_apply = [
        A.ToFloat(), #Converts from 0-255 to 0-1

        A.Resize( #Resizes to 160x160
            height=resize_after_crop,
            width=resize_after_crop,
            interpolation=1,
            always_apply=True
        ),

        #A.randomrotate
    ]
    return A.Compose(transforms_to_apply)

# ---------------------------------------------------------------------------------

def generate_datamodule(DATASET,MODE,PATHS,datasets,modes,IMG_SIZE,BATCH_SIZE,NUM_WORKERS):
    if DATASET == datasets.CUT_DATASET:
        train_val_catalog = pd.read_csv(PATHS["CUT_CATALOG_TRAIN_PATH"])
        train_val_catalog["file_loc"] = get_file_paths(train_val_catalog,PATHS["FULL_DATA_PATH"])
        generator1 = torch.Generator().manual_seed(42) #Preset test-val split, note test dataloader will still shuffle
        train_catalog, val_catalog = random_split(train_val_catalog, [0.20,0.80], generator=generator1)
        train_catalog = train_catalog.dataset.iloc[train_catalog.indices]
        val_catalog = val_catalog.dataset.iloc[val_catalog.indices]   
        test_catalog = pd.read_csv(PATHS["CUT_CATALOG_TEST_PATH"])
        test_catalog["file_loc"] = get_file_paths(test_catalog,PATHS["FULL_DATA_PATH"])

        datamodule = GalaxyDataModule(
            label_cols=["P_CW","P_ACW","P_OTHER"],
            train_catalog=train_catalog, val_catalog=train_catalog, test_catalog=test_catalog,
            custom_albumentation_transform=generate_transforms(resize_after_crop=IMG_SIZE),
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        )
    else:
        if DATASET == datasets.FULL_DATASET:
            catalog = pd.read_csv(PATHS["FULL_CATALOG_PATH"])
            catalog["file_loc"] = get_file_paths(catalog,PATHS["FULL_DATA_PATH"])

        if DATASET == datasets.FULL_DESI_DATASET:
            catalog = pd.read_csv(PATHS["FULL_DESI_CATALOG_PATH"])
            catalog["file_loc"] = get_file_paths(catalog,PATHS["FULL_DATA_PATH"])

        elif DATASET == datasets.BEST_SUBSET:
            catalog = pd.read_csv(PATHS["BEST_SUBSET_CATALOG_PATH"])
            catalog["file_loc"] = get_file_paths(catalog,PATHS["FULL_DATA_PATH"])

        elif DATASET == datasets.LOCAL_SUBSET:
            catalog = pd.read_csv(PATHS["LOCAL_SUBSET_CATALOG_PATH"])
            catalog["file_loc"] = get_file_paths(catalog,PATHS["LOCAL_SUBSET_DATA_PATH"])

        if MODE == modes.PREDICT:
            datamodule = GalaxyDataModule(
            label_cols=None,
            predict_catalog=catalog,
            custom_albumentation_transform=generate_transforms(resize_after_crop=IMG_SIZE),
            batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            )
        else:
            datamodule = GalaxyDataModule(
                label_cols=["P_CW","P_ACW","P_OTHER"],
                catalog=catalog,
                train_fraction=0.7, val_fraction=0.15, test_fraction=0.15,
                custom_albumentation_transform=generate_transforms(resize_after_crop=IMG_SIZE),
                batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
            )
    return datamodule

# ---------------------------------------------------------------------------------

def get_device():
    print(f"Using pytorch {torch.__version__}. CPU cores available on device: {os.cpu_count()}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print(torch.cuda.get_device_name(0))
        print(f"Allocated Memory: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(f"Cached Memory: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")
    print("Using device:", device)
    return device