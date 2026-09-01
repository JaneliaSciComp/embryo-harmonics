import argparse
import logging
import os
import time

from tqdm import tqdm

from embryoharmonics import celegans
from embryoharmonics.celegans import AxisymmetricHarmonics, save_axisymmetric_harmonics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate meridian meshes and axisymmetric harmonics for "
        "all (rotationally symmetrized) C. elegans embryo model time steps."
    )
    parser.add_argument("path", help="Path to celegans_models.h5")
    parser.add_argument(
        "--output-file",
        default=os.path.join(os.getcwd(), "results", "symmetric_embryo_harmonics.h5"),
        help="HDF5 file to write meridian meshes and harmonics to; a sibling "
        ".xdmf file is generated alongside it (default: %(default)s)",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=300,
        help="Number of harmonics to compute (default: %(default)s)",
    )
    parser.add_argument(
        "--mesh-size",
        type=float,
        default=5,
        help="Mesh size used to generate the meridian mesh (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("embryoharmonics")
    logger.setLevel(logging.INFO)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    handler = logging.FileHandler(
        os.path.join(log_dir, f"process_all_geometries_{timestr}.log")
    )
    formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Load the embryo model and process each time step
    embryo_model_loader = celegans.EmbryoModelLoader(args.path)
    output_file = os.path.normpath(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for time_step in tqdm(embryo_model_loader.time_steps):
        logger.info("Processing time step %d", time_step)
        embryo_model = embryo_model_loader.load(time_step, symmetric=True)
        mesh = embryo_model.generate_meridian_mesh(mesh_size=args.mesh_size)
        harmonics = AxisymmetricHarmonics.compute(mesh, n=args.n_harmonics)

        logger.info("Saving meridian mesh and harmonics to %s", output_file)
        save_axisymmetric_harmonics(output_file, time_step, harmonics)


if __name__ == "__main__":
    main()
