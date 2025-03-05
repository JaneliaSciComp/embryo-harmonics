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
import pyvista as pv
from tqdm.notebook import tqdm

from embryoharmonics import *
from embryoharmonics._utils import all_harmonic_names

# %%
# Set up logging
os.makedirs(os.path.join(os.getcwd(), '..', 'logs'), exist_ok=True)
logger = logging.getLogger("embryoharmonics")
logger.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")
handler = logging.FileHandler(os.path.join(os.getcwd(), '..', 'logs', f'{timestr}_process_all_genes.log'))
formatter = logging.Formatter('[%(asctime)s - %(name)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
PATH = "/Users/innerbergerm/Data/worm-geometry/4D_transcriptome.mat"
h5file = h5py.File(PATH, 'r')
gene_data_loader = GeneDataLoader(h5file)
time_steps = range(420, 841)

# %%
root = os.path.normpath(os.path.join(os.getcwd(), '..', 'results'))
os.makedirs(root, exist_ok=True)

# %%
# Find out at which time steps gene data is actually available
actual_time_steps = h5file['geneact/timepoints'][0]
time_steps = [t for t in time_steps if t in actual_time_steps]

# %%
# Find out how many harmonics were stored
# The order of the harmonics will match the order of the coefficients computed below
first_geometry = pv.read(os.path.join(root, "data_420.vtu"))
all_harmonics = all_harmonic_names(first_geometry)


# %%
def write_meta_data(file):
    # Write gene names (same format as matlab char arrays are stored in mat files: an array of space padded ascii-chars)
    max_gene_name_length = max(len(name) for name in gene_data_loader.gene_names)
    gene_names = [name.ljust(max_gene_name_length).encode('ascii') for name in gene_data_loader.gene_names]
    gene_names = np.array([np.frombuffer(name, dtype=np.uint8) for name in gene_names]).T.astype(np.uint16)
    file.create_dataset('gene_names', data=gene_names)

    # Write tissue names (same format as matlab char arrays are stored in mat files: an array of space padded ascii-chars)
    max_tissue_name_length = max(len(name) for name in gene_data_loader.tissue_names)
    tissue_names = [name.ljust(max_tissue_name_length).encode('ascii') for name in gene_data_loader.tissue_names]
    tissue_names = np.array([np.frombuffer(gene_names[:, i], dtype=np.uint8) for i in range(len(tissue_names))]).T.astype(np.uint16)
    file.create_dataset('tissue_names', data=tissue_names)

    # Write all time points
    time_points = np.array(list(time_steps))
    file.create_dataset('time_points', data=time_points)

    # Write harmonic names to match coefficients to harmonics
    harmonic_names = [name.encode('ascii') for name in all_harmonics]
    harmonic_names = np.array([np.frombuffer(name, dtype=np.uint8) for name in harmonic_names]).T.astype(np.uint16)
    file.create_dataset('harmonic_names', data=harmonic_names)


# %%
def write_data(file):
    gene_names = gene_data_loader.gene_names
    tissue_names = gene_data_loader.tissue_names

    # Set up arrays of coefficients to be filled
    gene_coeff = np.zeros((len(time_steps), len(gene_names), len(all_harmonics)), dtype=np.float64)
    tissue_coeff = np.zeros((len(time_steps), len(tissue_names), len(all_harmonics)), dtype=np.float64)

    for i, t in enumerate(tqdm(time_steps)):
        # Load the mesh for the current time step and compute coefficients for all genes and tissues
        pv_data = pv.read(os.path.join(root, f"data_{t:03d}.vtu"))

        gene_data = [gene_data_loader.load(gene, t) for gene in gene_names]
        tissue_data = [gene_data_loader.load_tissue(tissue, t) for tissue in tissue_names]
        eigen_coefficients = compute_harmonic_coefficients(pv_data, gene_data + tissue_data)

        # Sort coefficients into the preallocated arrays
        for j, name in enumerate(gene_names):
            gene_coeff[i, j, :] = eigen_coefficients[name]
        for j, name in enumerate(tissue_names):
            tissue_coeff[i, j, :] = eigen_coefficients[name]

    file.create_dataset('gene_coefficients', data=gene_coeff)
    file.create_dataset('tissue_coefficients', data=tissue_coeff)


# %%
# Execute and write everything
with h5py.File(os.path.join(root, 'eigen_coefficients.h5'), 'w') as target_file:
    write_meta_data(target_file)
    write_data(target_file)
