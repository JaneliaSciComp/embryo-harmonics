import csv
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
    ("ovoid-horn-phi", []),
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


def test_compare_approximations(tmp_path):
    """The comparison should score the exact harmonics perfectly and write a
    figure and a metrics table.
    """
    for folder, name in (("ovoid-symmetric", "exact"), ("ovoid-horn-phi", "horn_phi")):
        run_script(
            folder, "process_geometry.py", COEFFICIENT_FILE,
            "--output-file", tmp_path / f"{name}_harmonics.h5",
            "--n-harmonics", 10, "--mesh-size", 20, cwd=tmp_path,
        )
    run_script("ovoid-horn-phi", "compare_approximations.py", tmp_path, cwd=tmp_path)

    assert (tmp_path / "approximation_comparison.png").exists()
    with open(tmp_path / "approximation_metrics.csv") as file:
        rows = {row["variant"]: row for row in csv.DictReader(file)}
    assert float(rows["exact"]["mean_best_match_all"]) == pytest.approx(1.0)
    assert 0 < float(rows["horn_phi"]["mean_best_match_all"]) <= 1
