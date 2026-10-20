import argparse
import logging
import os
import time

from embryoharmonics.celegans import (
    AxisymmetricHarmonics,
    OvoidModel,
    save_axisymmetric_harmonics,
    save_ovoid,
)
from embryoharmonics.celegans.ovoid import DEFAULT_PIXEL_SIZE_UM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate the meridian mesh and axisymmetric harmonics of "
        "the ovoid (pre-twitch) embryo geometry. There is one geometry for all "
        "time points, stored under time point 0."
    )
    parser.add_argument("path", help="Path to the csv file of outline Fourier coefficients")
    parser.add_argument(
        "--output-file",
        default=os.path.join(os.getcwd(), "results", "ovoid_symmetric_harmonics.h5"),
        help="HDF5 file to write the meridian mesh and harmonics to; a sibling "
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
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=DEFAULT_PIXEL_SIZE_UM,
        metavar="UM",
        help="Pixel size of the outline coefficients in microns (default: %(default)s)",
    )
    parser.add_argument(
        "--radial-extension",
        type=float,
        default=0.0,
        metavar="UM",
        help="Push the outline out by this many microns before meshing, e.g. "
        "if the cells don't fit (default: %(default)s)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Point the first tip of the outline (theta = 0) to the negative "
        "instead of the positive AP direction",
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
    handler = logging.FileHandler(os.path.join(log_dir, f"process_geometry_{timestr}.log"))
    formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    ovoid = OvoidModel.from_csv(
        args.path,
        pixel_size=args.pixel_size,
        radial_extension=args.radial_extension,
        flip=args.flip,
    )
    output_file = os.path.normpath(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    mesh = ovoid.generate_meridian_mesh(mesh_size=args.mesh_size)
    harmonics = AxisymmetricHarmonics.compute(mesh, n=args.n_harmonics)

    logger.info("Saving meridian mesh and harmonics to %s", output_file)
    save_axisymmetric_harmonics(output_file, 0, harmonics)
    save_ovoid(output_file, ovoid)


if __name__ == "__main__":
    main()
