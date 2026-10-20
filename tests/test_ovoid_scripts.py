import os
import subprocess
import sys

import h5py
import numpy as np
import pytest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COEFFICIENT_FILE = os.path.join(ROOT, "tests", "resources", "ovoid_coefficients.csv")
GENE_PATH = os.path.join(ROOT, "tests", "resources", "embryo_pre_twitch.zarr")


def run_script(folder, name, *args, cwd):
    """Run a script from scripts/<folder> with the given arguments."""
    script = os.path.join(ROOT, "scripts", folder, name)
    subprocess.run([sys.executable, script, *map(str, args)], cwd=cwd, check=True)


@pytest.mark.parametrize("folder, extra_args", [
    ("ovoid-symmetric", []),
    ("ovoid-horn-z", ["--caps", "natural"]),
    ("ovoid-horn-z", ["--caps", "flat"]),
    ("ovoid-horn-z", ["--caps", "corrected"]),
])
def test_ovoid_pipeline_end_to_end(tmp_path, folder, extra_args):
    """Geometry and gene processing should yield finite coefficients for all
    test time points, with all cells inside the (default) ovoid.
    """
    harmonics_file = tmp_path / "harmonics.h5"
    run_script(
        folder, "process_geometry.py", COEFFICIENT_FILE,
        "--output-file", harmonics_file, "--n-harmonics", 10, "--mesh-size", 20,
        *extra_args, cwd=tmp_path,
    )
    run_script(
        folder, "process_all_genes.py", GENE_PATH, harmonics_file,
        "--output-file", "coefficients.h5", cwd=tmp_path,
    )

    with h5py.File(tmp_path / "coefficients.h5", "r") as file:
        coefficients = file["gene_coefficients"][:]
        assert list(file["time_points"][:]) == [100, 202, 362]
        assert coefficients.shape[:2] == (3, 5)
        assert coefficients.shape[2] >= 10
        assert np.all(np.isfinite(coefficients))
        assert file["max_relative_radius"][()] <= 1
