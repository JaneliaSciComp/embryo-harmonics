import argparse
import logging
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import h5py
from tqdm import tqdm

from embryoharmonics import celegans, io
from embryoharmonics import Harmonics


logger = logging.getLogger("embryoharmonics")

# Netgen fails on a few quasi-random (time step, mesh size) combinations, some
# with uncatchable native aborts (double free). Therefore, each time step runs
# in its own subprocess and is retried with a nudged mesh size on failure.
MESH_SIZE_NUDGE = 0.05


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate meshes and harmonics for all C. elegans embryo model time steps."
    )
    parser.add_argument("path", help="Path to celegans_models.h5")
    parser.add_argument(
        "--output-file",
        default=os.path.join(os.getcwd(), "results", "embryo_harmonics.h5"),
        help="HDF5 file to write meshes and harmonics to; a sibling .xdmf file "
        "is generated alongside it (default: %(default)s)",
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
        default=4.75,
        help="Mesh size used to generate the embryo mesh (default: %(default)s)",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=8,
        help="Number of time steps to process in parallel (default: %(default)s)",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=3,
        help="Attempts per time step, each nudging the mesh size by "
        f"+{MESH_SIZE_NUDGE} (default: %(default)s)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help="Timeout in seconds per attempt (default: %(default)s)",
    )
    parser.add_argument("--worker", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def process_one(args):
    """Process a single time step (runs in a subprocess); crashes only kill
    this process, and the parent retries with a nudged mesh size.
    """
    time_step = args.worker
    embryo_model = celegans.EmbryoModelLoader(args.path).load(time_step)
    mesh = embryo_model.generate_mesh(mesh_size=args.mesh_size)
    harmonics = Harmonics.compute(mesh, n=args.n_harmonics)
    io.save_time_point(args.output_file, time_step, mesh, harmonics)


def run_with_retries(args, time_step):
    """Run one time step in a subprocess, retrying with nudged mesh sizes.

    :return: The temporary HDF5 file with the results, or None if all
        attempts failed.
    """
    temp_file = f"{args.output_file[:-3]}.t{time_step:03d}.h5"
    for attempt in range(args.max_attempts):
        mesh_size = args.mesh_size + attempt * MESH_SIZE_NUDGE
        cmd = [
            sys.executable, os.path.abspath(__file__), args.path,
            "--worker", str(time_step),
            "--output-file", temp_file,
            "--mesh-size", str(mesh_size),
            "--n-harmonics", str(args.n_harmonics),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Time step %d timed out at mesh size %g", time_step, mesh_size)
            continue

        if result.returncode == 0:
            if attempt > 0:
                logger.info("Time step %d succeeded at nudged mesh size %g", time_step, mesh_size)
            return temp_file
        logger.warning(
            "Time step %d failed at mesh size %g:\n%s",
            time_step, mesh_size, result.stderr.strip()
        )
    return None


def merge_time_point(output_file, temp_file, time_step):
    """Move one time step's results from its temporary file into the output file."""
    key = f"{time_step:03d}"
    with h5py.File(temp_file, "r") as src, h5py.File(output_file, "a") as dst:
        for group in ("meshes", "harmonics"):
            parent = dst.require_group(group)
            if key in parent:
                del parent[key]
            dst.copy(src[f"{group}/{key}"], parent, name=key)
    os.remove(temp_file)
    xdmf_sibling = os.path.splitext(temp_file)[0] + ".xdmf"
    if os.path.exists(xdmf_sibling):
        os.remove(xdmf_sibling)


def main():
    args = parse_args()

    if args.worker is not None:
        process_one(args)
        return

    # Set up logging
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    logger.setLevel(logging.INFO)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    handler = logging.FileHandler(
        os.path.join(log_dir, f"process_all_geometries_{timestr}.log")
    )
    formatter = logging.Formatter("[%(asctime)s - %(name)s] %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # Process all time steps in parallel subprocesses; only the main thread
    # writes to the output file (HDF5 allows a single writer)
    embryo_model_loader = celegans.EmbryoModelLoader(args.path)
    output_file = os.path.normpath(args.output_file)
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    time_steps = embryo_model_loader.time_steps
    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(run_with_retries, args, time_step): time_step
            for time_step in time_steps
        }
        for future in tqdm(as_completed(futures), total=len(futures)):
            time_step = futures[future]
            temp_file = future.result()
            if temp_file is None:
                failed.append(time_step)
                continue
            logger.info("Saving mesh and harmonics for time step %d to %s",
                        time_step, output_file)
            merge_time_point(output_file, temp_file, time_step)

    io._write_xdmf(output_file)

    if failed:
        logger.error("Failed time steps: %s", sorted(failed))
        sys.exit(f"Failed time steps: {sorted(failed)}")


if __name__ == "__main__":
    main()
