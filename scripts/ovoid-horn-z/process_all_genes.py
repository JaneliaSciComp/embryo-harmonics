import argparse
import logging
import os
import time

import h5py
import numpy as np
from tqdm import tqdm

from embryoharmonics import celegans, io
from embryoharmonics.celegans import load_axisymmetric_harmonics, load_ovoid


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute harmonic coefficients for all genes and tissues on "
        "the ovoid (pre-twitch) geometry. The cells of all time points are "
        "shifted by one translation that fits them into the ovoid; the modes "
        "are evaluated at the cell positions directly from the stored meridian "
        "fields."
    )
    parser.add_argument("gene_path", help="Path to the gene data: an HDF5 file, a directory of parquet files or a zarr store")
    parser.add_argument(
        "result_path",
        help="Path to the HDF5 file containing the meridian mesh, harmonics "
        "and ovoid outline",
    )
    parser.add_argument(
        "--output-file",
        default="harmonic_coefficients.h5",
        help="Output HDF5 file; a bare file name is written next to result_path "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--last-time-step",
        type=int,
        default=380,
        help="Only process gene data up to this time step, i.e. before the "
        "embryo starts twitching (default: %(default)s)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        nargs="+",
        default=[0.0],
        metavar="PERCENT",
        help="Scale the cell positions up by this percentage so that they fit "
        "into a slightly too small geometry: one value for all directions or "
        "two values for the radial (x, y) and axial (z) direction (default: %(default)s)",
    )
    return parser.parse_args()


def write_meta_data(file, gene_data_loader, time_steps, harmonics, center, fill):
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
    harmonic_names = [f"harmonic_{i:03d}" for i in range(len(harmonics))]
    file.create_dataset("harmonic_names", data=io.encode_matlab_strings(harmonic_names))
    file.create_dataset("angular_orders", data=harmonics.angular_orders)
    file.create_dataset("trig_kinds", data=harmonics.trig_kinds.astype("S3"))

    # The position (in voxels) moved to the ovoid's center, and how far out
    # the outermost cell lies relative to the outline (> 1 means outside)
    file.create_dataset("center", data=center)
    file.create_dataset("max_relative_radius", data=fill)


def write_data(file, gene_data_loader, harmonics, time_steps, center, logger):
    """Write the harmonic coefficients for all genes and tissues to the given HDF5 file."""
    # Set up arrays of coefficients to be filled
    gdl = gene_data_loader
    gene_coeff = np.zeros((len(time_steps), gdl.n_genes, len(harmonics)), dtype=np.float64)
    tissue_coeff = np.zeros((len(time_steps), gdl.n_tissues, len(harmonics)), dtype=np.float64)
    logger.info(
        "Preallocated arrays for %d time steps, %d genes and %d tissues",
        len(time_steps),
        gdl.n_genes,
        gdl.n_tissues,
    )

    for i, t in enumerate(tqdm(time_steps)):
        # Decompose all genes and tissues at once, skipping the interpolation
        logger.info("Processing time step %d", t)
        locations, activities = gdl.load_all(t)
        coefficients = harmonics.decompose_point_data(locations - center, activities)
        gene_coeff[i] = coefficients[:gdl.n_genes]
        tissue_coeff[i] = coefficients[gdl.n_genes:]

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

    gene_data_loader = celegans.open_gene_data_loader(
        args.gene_path, location_scale=celegans.gene_data.buffered_location_scale(args.buffer)
    )
    time_steps = [t for t in gene_data_loader.time_steps if t <= args.last_time_step]
    if len(time_steps) == 0:
        raise ValueError(f"No gene data up to time step {args.last_time_step}!")

    # One geometry for all time steps, stored under time point 0
    harmonics = load_axisymmetric_harmonics(args.result_path, 0)
    ovoid = load_ovoid(args.result_path)

    # Fit one translation for the cells of all time steps
    all_locations = np.vstack([gene_data_loader.load_all(t)[0] for t in time_steps])
    center = ovoid.fit_center(all_locations)
    fill = ovoid.containment(all_locations, center).max()
    if fill > 1:
        logger.warning(
            "Some cells lie outside the ovoid (relative radius %.4f); consider "
            "a radial extension of the geometry", fill
        )
    logger.info(
        "Compute coefficients for %d time steps and %d harmonics with center %s",
        len(time_steps), len(harmonics), center,
    )

    # Execute and write everything
    target_file_name = args.output_file
    if not os.path.dirname(target_file_name):
        target_file_name = os.path.join(os.path.dirname(args.result_path), target_file_name)
    with h5py.File(target_file_name, "w") as target_file:
        logger.info("Write data to %s", target_file_name)
        write_meta_data(target_file, gene_data_loader, time_steps, harmonics, center, fill)
        write_data(target_file, gene_data_loader, harmonics, time_steps, center, logger)


if __name__ == "__main__":
    main()
