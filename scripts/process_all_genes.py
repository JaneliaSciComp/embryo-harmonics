# %%
import logging
import os
import time
import re

import h5py
import numpy as np
from tqdm import tqdm

from embryoharmonics import celegans, io
from embryoharmonics import encode_matlab_strings

# %%
# Set up logging
log_dir = os.path.join(os.getcwd(), 'logs')
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("embryoharmonics")
logger.setLevel(logging.INFO)
timestr = time.strftime("%Y%m%d-%H%M%S")
handler = logging.FileHandler(os.path.join(log_dir, f'process_all_genes_{timestr}.log'))
formatter = logging.Formatter('[%(asctime)s - %(name)s] %(levelname)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# %%
GENE_PATH = "<your_path_to_celegans_genedata.h5>"
RESULT_PATH = "<your_path_to_meshes_and_harmonics>"
gene_data_loader = celegans.GeneDataLoader(GENE_PATH)

# %%
def find_and_sort_files(directory, regexp):
    """Find files matching a regular expression in a directory and extract time steps from them.
    """
    file_pattern = re.compile(regexp)
    pairs = [(file, int(match.group(1)))
             for file in os.listdir(directory)
             if (match := file_pattern.match(file)) is not None]

    # Sort both lists based on the extracted numbers
    sorted_pairs = sorted(pairs, key=lambda pair: pair[1])
    return zip(*sorted_pairs) if sorted_pairs else ([], [])

# Discover mesh and harmonics files and make sure the times match
mesh_files, mesh_times = find_and_sort_files(RESULT_PATH, r"data_(\d+)\.vtu")
harmonic_files, harmonic_times = find_and_sort_files(RESULT_PATH, r"harmonics_(\d+)\.h5")
if len(mesh_files) == 0 or len(harmonic_files) == 0:
    raise ValueError("No mesh or harmonics files found in the given directory!")
if mesh_times != harmonic_times:
    raise ValueError("Mesh and harmonics files do not match!")
logger.info("Found %d mesh and harmonics files", len(mesh_files))

# %%
# Find out at which time steps gene data is actually available and how many harmonics we have
time_steps = [t for t in mesh_times if t in gene_data_loader.time_steps]
if len(time_steps) == 0:
    raise ValueError("No matching time steps found between mesh files and gene data!")

first_mesh = io.load_mesh(os.path.join(RESULT_PATH, mesh_files[0]))
first_harmonics = io.load_harmonics(os.path.join(RESULT_PATH, harmonic_files[0]), first_mesh)
n_harmonics = len(first_harmonics)
logger.info("Compute coefficients for %d times steps and  %d harmonics",
            len(time_steps), n_harmonics)


# %%
def write_meta_data(file):
    """Write metadata about the harmonic coefficients to the given HDF5 file."""
    # Write gene and tissue names
    file.create_dataset('gene_names', data=encode_matlab_strings(gene_data_loader.gene_names))
    file.create_dataset('tissue_names', data=encode_matlab_strings(gene_data_loader.tissue_names))

    # Write all time points
    time_points = np.array(time_steps)
    file.create_dataset('time_points', data=time_points)

    # Write harmonic names to match coefficients to harmonics
    harmonic_names = [f"harmonic_{i:03d}" for i in range(n_harmonics)]
    file.create_dataset('harmonic_names', data=encode_matlab_strings(harmonic_names))


# %%
def write_data(file):
    """Write the harmonic coefficients for all genes and tissues to the given HDF5 file."""
    # Set up arrays of coefficients to be filled
    gdl = gene_data_loader
    gene_coeff = np.zeros((gdl.n_time_steps, gdl.n_genes, n_harmonics), dtype=np.float64)
    tissue_coeff = np.zeros((gdl.n_time_steps, gdl.n_tissues, n_harmonics), dtype=np.float64)
    logger.info("Preallocated arrays for %d time steps, %d genes and %d tissues",
                gdl.n_time_steps, gdl.n_genes, gdl.n_tissues)

    for i, t in enumerate(tqdm(time_steps)):
        # Load the current time step and compute coefficients for all genes and tissues
        logger.info("Processing time step %d", t)
        mesh = io.load_mesh(os.path.join(RESULT_PATH, mesh_files[i]))
        harmonics = io.load_harmonics(os.path.join(RESULT_PATH, harmonic_files[i]), mesh)

        gene_data = [gdl.load(gene, t).interpolate(mesh)
                     for gene in gdl.gene_names]
        tissue_data = [gdl.load_tissue(tissue, t).interpolate(mesh)
                       for tissue in gdl.tissue_names]
        eigen_coefficients = harmonics.decompose(gene_data + tissue_data)

        # Sort coefficients into the preallocated arrays
        for j, name in enumerate(gdl.gene_names):
            gene_coeff[i, j, :] = eigen_coefficients[name]
        for j, name in enumerate(gdl.tissue_names):
            tissue_coeff[i, j, :] = eigen_coefficients[name]

    logger.info("Write %d gene and %d tissue coefficients to disk", gdl.n_genes, gdl.n_tissues)
    file.create_dataset('gene_coefficients', data=gene_coeff)
    file.create_dataset('tissue_coefficients', data=tissue_coeff)


# %%
# Execute and write everything
target_file_name = os.path.join(RESULT_PATH, 'harmonic_coefficients.h5')
with h5py.File(target_file_name, 'w') as target_file:
    logger.info("Write data to %s", target_file_name)
    write_meta_data(target_file)
    write_data(target_file)

# %%
