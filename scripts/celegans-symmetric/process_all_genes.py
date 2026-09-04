import argparse
import logging
import os
import time

import h5py
import numpy as np
from tqdm import tqdm

from embryoharmonics import celegans, io
from embryoharmonics.celegans import load_axisymmetric_harmonics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute axisymmetric harmonic coefficients for all genes "
        "and tissues. The 3D basis is reconstructed per time step by revolving "
        "the stored meridian mesh."
    )
    parser.add_argument("gene_path", help="Path to the gene data: an HDF5 file or a directory of parquet files")
    parser.add_argument(
        "result_path",
        help="Path to the HDF5 file containing meridian meshes and "
        "axisymmetric harmonics",
    )
    parser.add_argument(
        "--output-file",
        default="harmonic_coefficients.h5",
        help="Name of the output HDF5 file, written next to result_path "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=5,
        help="Mesh size used to generate the revolved 3D mesh (default: %(default)s)",
    )
    return parser.parse_args()


def write_meta_data(file, gene_data_loader, time_steps, first_harmonics, n_harmonics):
    """Write metadata about the harmonic coefficients to the given HDF5 file."""
    # Write gene and tissue names
    file.create_dataset(
        "gene_names", data=io.encode_matlab_strings(gene_data_loader.gene_names)
    )
    file.create_dataset(
        "tissue_names", data=io.encode_matlab_strings(gene_data_loader.tissue_names)
    )

    # Write all time points
    time_points = np.array(time_steps)
    file.create_dataset("time_points", data=time_points)

    # Write harmonic names and angular labels to match coefficients to harmonics
    harmonic_names = [f"harmonic_{i:03d}" for i in range(n_harmonics)]
    file.create_dataset("harmonic_names", data=io.encode_matlab_strings(harmonic_names))
    file.create_dataset(
        "angular_orders", data=first_harmonics.angular_orders[:n_harmonics]
    )
    file.create_dataset(
        "trig_kinds", data=first_harmonics.trig_kinds[:n_harmonics].astype("S3")
    )


def write_data(
    file,
    gene_data_loader,
    result_path,
    time_steps,
    n_harmonics,
    mesh_size,
    logger,
):
    """Write the harmonic coefficients for all genes and tissues to the given HDF5 file."""
    # Set up arrays of coefficients to be filled
    gdl = gene_data_loader
    gene_coeff = np.zeros((len(time_steps), gdl.n_genes, n_harmonics), dtype=np.float64)
    tissue_coeff = np.zeros((len(time_steps), gdl.n_tissues, n_harmonics), dtype=np.float64)
    logger.info(
        "Preallocated arrays for %d time steps, %d genes and %d tissues",
        len(time_steps),
        gdl.n_genes,
        gdl.n_tissues,
    )

    for i, t in enumerate(tqdm(time_steps)):
        # Load the current time step and reconstruct the 3D basis
        logger.info("Processing time step %d", t)
        axisymmetric = load_axisymmetric_harmonics(result_path, t)
        harmonics = axisymmetric.to_full_3d(mesh_size=mesh_size)
        mesh = harmonics.mesh

        # The keep-pairs truncation rule can yield n or n + 1 modes per step
        m = min(n_harmonics, len(harmonics))
        if m < len(harmonics):
            logger.warning(
                "Time step %d has %d harmonics; keeping the first %d",
                t, len(harmonics), m,
            )

        # Don't remove nans to optimize internal caching of location lookup
        gene_data = [
            gdl.load(gene, t, remove_nans=False).interpolate(mesh)
            for gene in gdl.gene_names
        ]
        tissue_data = [
            gdl.load_tissue(tissue, t).interpolate(mesh) for tissue in gdl.tissue_names
        ]
        eigen_coefficients = harmonics.decompose(gene_data + tissue_data)

        # Sort coefficients into the preallocated arrays
        for j, name in enumerate(gdl.gene_names):
            gene_coeff[i, j, :m] = eigen_coefficients[name][:m]
        for j, name in enumerate(gdl.tissue_names):
            tissue_coeff[i, j, :m] = eigen_coefficients[name][:m]

    logger.info(
        "Write %d gene and %d tissue coefficients to disk", gdl.n_genes, gdl.n_tissues
    )
    file.create_dataset("gene_coefficients", data=gene_coeff)
    file.create_dataset("tissue_coefficients", data=tissue_coeff)


def main():
    args = parse_args()

    # Set up logging
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("embryoharmonics")
    logger.setLevel(logging.INFO)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    handler = logging.FileHandler(
        os.path.join(log_dir, f"process_all_genes_{timestr}.log")
    )
    formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    gene_data_loader = celegans.open_gene_data_loader(args.gene_path)

    # Discover time points with meshes and harmonics
    result_times = io.time_points(args.result_path)
    logger.info("Found %d time points with meshes and harmonics", len(result_times))

    # Find out at which time steps gene data is actually available and how many harmonics we have
    time_steps = [t for t in result_times if t in gene_data_loader.time_steps]
    if len(time_steps) == 0:
        raise ValueError(
            "No matching time steps found between result file and gene data!"
        )

    first_harmonics = load_axisymmetric_harmonics(args.result_path, time_steps[0])
    n_harmonics = len(first_harmonics)
    logger.info(
        "Compute coefficients for %d times steps and %d harmonics",
        len(time_steps),
        n_harmonics,
    )

    # Execute and write everything
    target_file_name = os.path.join(os.path.dirname(args.result_path), args.output_file)
    with h5py.File(target_file_name, "w") as target_file:
        logger.info("Write data to %s", target_file_name)
        write_meta_data(
            target_file, gene_data_loader, time_steps, first_harmonics, n_harmonics
        )
        write_data(
            target_file,
            gene_data_loader,
            args.result_path,
            time_steps,
            n_harmonics,
            args.mesh_size,
            logger,
        )


if __name__ == "__main__":
    main()
