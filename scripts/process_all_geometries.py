# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
import logging
import os
import time

import h5py
import numpy as np
from tqdm.notebook import tqdm

from embryoharmonics import *

# %%
# Set up logging
logger = logging.getLogger("embryoharmonics")
logger.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")
handler = logging.FileHandler(os.path.join(os.getcwd(), '..', 'logs', f'{timestr}_process_all_geometries.log'))
formatter = logging.Formatter('[%(asctime)s - %(name)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
path = "/home/innerbergerm@hhmi.org/big-data/worm-geometry/celegans_avg_models_2024_04_23.h5"
h5file = h5py.File(path, 'r')

# %%
k = 100
time_steps = range(420, 621, 5)
embryo_model_loader = EmbryoModelLoader(h5file)
root = os.path.normpath(os.path.join(os.getcwd(), '..', 'results'))
os.makedirs(root, exist_ok=True)

# %%
for time_step in tqdm(time_steps):
    embryo_model = embryo_model_loader.load(time_step)
    mesh = generate_embryo_mesh(embryo_model, mesh_size=5)
    pv_data, metrics = compute_harmonics(mesh, k=k, boundary_condition="neumann", store_dirichlet_densities=True)

    data_file_name = os.path.join(root, f"data_{time_step:03d}.vtu")
    pv_data.save(data_file_name)

    metrics_file_name = os.path.join(root, f"metrics_{time_step:03d}.csv")
    values = np.column_stack(list(metrics.values()))
    np.savetxt(metrics_file_name, values, delimiter=",", header=",".join(metrics.keys()))

# %%
