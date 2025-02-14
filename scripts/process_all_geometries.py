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
os.makedirs(os.path.join(os.getcwd(), '..', 'logs'), exist_ok=True)
logger = logging.getLogger("embryoharmonics")
logger.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")
handler = logging.FileHandler(os.path.join(os.getcwd(), '..', 'logs', f'{timestr}_process_all_geometries.log'))
formatter = logging.Formatter('[%(asctime)s - %(name)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
PATH = "/Users/innerbergerm/Data/worm-geometry/celegans_avg_models_421_minutes_samples_2024_12_04.h5"
h5file = h5py.File(PATH, 'r')

# %%
k = 300
embryo_model_loader = EmbryoModelLoader(h5file)
root = os.path.normpath(os.path.join(os.getcwd(), '..', 'results'))
os.makedirs(root, exist_ok=True)

# %%
for time_step in tqdm(embryo_model_loader.time_steps):
    logger.info("Processing time step %d", time_step)
    embryo_model = embryo_model_loader.load(time_step)
    mesh = generate_embryo_mesh(embryo_model, mesh_size=5)
    pv_data, metrics = compute_harmonics(mesh, k=k, boundary_condition="neumann", store_dirichlet_densities=True)

    data_file_name = os.path.join(root, f"data_{time_step:03d}.vtu")
    pv_data.save(data_file_name)

    metrics_file_name = os.path.join(root, f"metrics_{time_step:03d}.csv")
    values = np.column_stack(list(metrics.values()))
    np.savetxt(metrics_file_name, values, delimiter=",", header=",".join(metrics.keys()))

# %%
