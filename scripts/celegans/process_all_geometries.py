import argparse
import logging
import os
import time

from tqdm import tqdm

from embryoharmonics import celegans, io
from embryoharmonics import Harmonics


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate meshes and harmonics for all C. elegans embryo model time steps."
    )
    parser.add_argument("path", help="Path to celegans_models.h5")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(os.getcwd(), "results"),
        help="Directory to write meshes and harmonics to (default: %(default)s)",
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
        help="Mesh size used to generate the embryo mesh (default: %(default)s)",
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
    root = os.path.normpath(args.output_dir)
    os.makedirs(root, exist_ok=True)

    for time_step in tqdm(embryo_model_loader.time_steps):
        logger.info("Processing time step %d", time_step)
        embryo_model = embryo_model_loader.load(time_step)
        mesh = embryo_model.generate_mesh(mesh_size=args.mesh_size)
        harmonics = Harmonics.compute(mesh, n=args.n_harmonics)

        mesh_path = os.path.join(root, f"data_{time_step:03d}.vtu")
        logger.info("Saving mesh to %s", mesh_path)
        io.save_mesh(mesh_path, mesh)

        harmonics_path = os.path.join(root, f"harmonics_{time_step:03d}.h5")
        logger.info("Saving harmonics to %s", harmonics_path)
        io.save_harmonics(harmonics_path, harmonics)


if __name__ == "__main__":
    main()
