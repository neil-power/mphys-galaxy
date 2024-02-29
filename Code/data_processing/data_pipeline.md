# Data Pre-processing

## Cross-matching DESI DR8 & GZ1 (SDSS DR7)
DESI images are organised by dr8_id, whereas GZ1 uses SDSS OBJID. Use astropy to match objects across both catalogs & add a 'dr8_id' column to the 

DESI catalog: Data/gz_desi_deep_learning_catalog_friendly.parquet

GZ1 catalog: Data/GalaxyZoo1_DR_table2.csv

Combined catalog (using Code/data_processing/dataset_manipulation.ipynb): Data/gz1_desi_cross_cat.csv

## Creating local subset of 1500 images
Local subset copied to Data/Subset for initial/local testing

Subset catalog (using Code/data_processing/create_data_subset.ipynb): Data/subset_gz1_desi_cross_cat.csv

## Cut objects that have another object within 1 arcsec
Query SDSS via astroquery to get r-band values, and cut objects that have another object within 1 arcsec

Queried catalog (using Code/data_processing/astroquery_batch.ipynb): Data/gz1_desi_cross_cat_queried.csv

## Cut objects using magnitude/r-band
Cut objects based on cuts made in Jia et al (2023)

Cut catalog (using Code/data_processing/data_cuts.ipynb): Data/gz1_desi_cross_cat_cut.csv