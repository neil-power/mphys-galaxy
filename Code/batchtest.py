import os
import pandas as pd
import numpy as np

desi_path = '/share/nas2/walml/galaxy_zoo/decals/dr8/jpg'

catalog = pd.read_csv('Data\gz1_desi_cross_cat.csv')

def get_paths(catalog):
    dr8_ids = catalog['dr8_id'].astype('str')
    brick_ids = dr8_ids.str.split('_', n=1,expand=True)[1]
    file_paths = desi_path+"/"+brick_ids+"/"+dr8_ids+".jpg"
    return file_paths

file_paths = get_paths(catalog)

results = []
for i in range(10):
    exists = os.path.isfile(file_paths[i])
    print(file_paths[i], exists)
    results.append(f"{file_paths[i]} ({exists})")

results = np.array(results).astype('str')
np.savetxt('results.csv',results,delimiter=",",fmt='%s')
