"""Webster-Horn accuracy across the whole developmental time course.

Runs the one-to-one eigenfunction comparison of
scripts/celegans/compare_webster_horn.py at every time step, and shows the
resulting scalar products as an image (time step vs mode index) for three
comparisons: Webster-Horn against the raw geometry's FEM basis, against the
rotationally symmetric geometry's FEM basis, and the two FEM bases against each
other (symmetrization alone, no Webster-Horn involved).

Caveat on that third comparison: the two FEM bases live on different meshes, so
unlike the Webster-Horn modes -- which are analytic and can be evaluated on
either mesh directly -- one of them has to be resampled onto the other. Its
scalar products therefore carry an interpolation error that the other two do
not, and should be read as a lower bound on agreement rather than an exact
symmetrization error.

The full sweep takes a few hours; time steps run in parallel worker processes
(N_WORKERS, default up to 8 cores -- override with the N_WORKERS env var). A
time step that fails, including one that crashes its worker process outright,
is left blank (it shows as a white column) rather than aborting the run.
"""
import multiprocessing as mp
import os
import warnings
from queue import Empty

import numpy as np
import matplotlib.pyplot as plt
import threadpoolctl
from tqdm import tqdm

from embryoharmonics import Harmonics, celegans
from embryoharmonics.celegans.webster_horn_harmonics import (
    compute_webster_horn_harmonics,
    degenerate_clusters,
    match_modes,
    orthonormalize_clusters,
)
from embryoharmonics.fem import FemMatrices

CWD = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CWD, "..", "..", "data", "avg_models_n371.h5")
RESULTS_DIR = os.path.join(CWD, "..", "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N = 100
N_POOL = N + 20  # oversized pool, so even the last FEM mode still has a partner
MESH_SIZE = 5
V_MIN = 0.995  # scalar products below this are clipped in the plot
# One process per time step: a bad geometry can segfault/double-free inside
# netgen/ngsolve, which no try/except can catch. Isolating each step in its own
# (spawned, not forked -- MKL/OpenMP thread state doesn't survive fork) process
# means that only costs the one step instead of the whole multi-hour sweep.
N_WORKERS = int(os.environ.get("N_WORKERS", min(os.cpu_count() or 4, 8)))
# Each worker's MKL/OpenMP calls default to using all cores, so N_WORKERS of
# them fight over the machine; cap each to its fair share of cores instead.
# MKL/OpenMP's own thread-team coordination across many concurrent processes is
# itself a plausible source of native crashes -- if those show up more often at
# higher N_WORKERS, try THREADS_PER_WORKER=1 to take MKL threading out of the
# picture entirely and rely on process-level parallelism alone.
THREADS_PER_WORKER = int(os.environ.get(
    "THREADS_PER_WORKER", max((os.cpu_count() or 1) // N_WORKERS, 1)))

# The mass-weighted matmuls raise spurious FP warnings on some BLAS backends
# (see compute_webster_horn_harmonics); the results are verified finite.
warnings.filterwarnings("ignore", category=RuntimeWarning)

model_loader = celegans.EmbryoModelLoader(MODEL_PATH)
time_steps = np.array(sorted(model_loader.time_steps))


FIELDS = ("product_raw", "product_sym", "product_fem")


def match(basis, targets, clusters):
    """Pair each target mode with the best-matching eigenfunction of the basis.

    :return: The matched cluster per target mode, and the scalar product with it
    """
    coefficients = basis.decompose(targets)
    matrix = np.column_stack([coefficients[data.name] for data in targets])
    return match_modes(matrix, clusters)


def scalar_products(time_step):
    """Scalar products for the three comparisons at one time step."""
    model_sym = model_loader.load(time_step, symmetric=True)
    mesh_sym = model_sym.generate_mesh(mesh_size=MESH_SIZE)
    webster, labels = compute_webster_horn_harmonics(mesh_sym, model_sym, n=N_POOL)
    clusters = degenerate_clusters(labels)
    harmonics_sym = Harmonics.compute(mesh_sym, n=N)
    targets_sym = [harmonics_sym[k] for k in range(N)]

    # The raw mesh is stretched onto the symmetric one; the Webster-Horn modes
    # are analytic, so they are evaluated directly on the stretched mesh rather
    # than resampled, which would add interpolation error.
    mesh_raw = model_loader.load(time_step).generate_mesh(mesh_size=MESH_SIZE)
    mesh_raw_stretched = celegans.transform_mesh(mesh_raw, mesh_sym)
    webster_raw, _ = compute_webster_horn_harmonics(
        mesh_raw_stretched, model_sym, n=N_POOL
    )
    harmonics_raw = Harmonics.compute(mesh_raw_stretched, n=N)
    targets_raw = [harmonics_raw[k] for k in range(N)]

    matched_sym, product_sym = match(webster, targets_sym, clusters)
    _, product_raw = match(webster_raw, targets_raw, clusters)

    # Symmetrization alone: the symmetric geometry's own FEM basis against the
    # raw one's. The degenerate structure of a FEM basis is not known from mode
    # labels, so it is taken from how the Webster-Horn clusters matched the
    # symmetric basis -- the pairing that compare_webster_horn.py checks is
    # one-to-one.
    fem_clusters = [
        list(np.flatnonzero(matched_sym == cluster))
        for cluster in np.unique(matched_sym)
    ]
    resampled = np.array([
        data.resample(mesh_raw_stretched, project_outside_data=True).data
        for data in targets_sym
    ])
    # Resampling destroys mass-normalization and within-cluster orthogonality,
    # both of which match_modes relies on.
    mass_raw = FemMatrices.compute_for(mesh_raw_stretched, stiffness=False).mass
    orthonormalize_clusters(resampled, fem_clusters, mass_raw)
    harmonics_sym_on_raw = Harmonics(
        mesh_raw_stretched, resampled, harmonics_sym.eigenvalues
    )
    _, product_fem = match(harmonics_sym_on_raw, targets_raw, fem_clusters)

    return product_raw, product_sym, product_fem


def _run_time_step(time_step, queue):
    """Runs one time step and reports back; a crash here only kills this process."""
    try:
        with threadpoolctl.threadpool_limits(limits=THREADS_PER_WORKER):
            result = scalar_products(time_step)
    except Exception as error:  # one awkward geometry should not kill the sweep
        queue.put((time_step, None, str(error)))
    else:
        queue.put((time_step, result, None))


if __name__ == "__main__":
    print(f"{len(time_steps)} time steps ({time_steps[0]}..{time_steps[-1]}) to compute, "
          f"{N_WORKERS} in parallel")

    # NaN marks a time step that failed and shows as a blank column in the plot.
    products = {name: np.full((len(time_steps), N), np.nan) for name in FIELDS}
    index_of = {int(step): i for i, step in enumerate(time_steps)}

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    pending = [int(step) for step in time_steps]
    running = {}

    def launch():
        if pending:
            step = pending.pop(0)
            process = ctx.Process(target=_run_time_step, args=(step, result_queue))
            process.start()
            running[step] = process

    for _ in range(min(N_WORKERS, len(pending))):
        launch()

    progress = tqdm(total=len(time_steps), unit="step", desc="time steps")
    while running:
        try:
            time_step, result, error = result_queue.get(timeout=1.0)
        except Empty:
            # A crashed process exits without ever putting a result; a clean
            # one always puts its result before exiting, so exitcode 0 here
            # just means its message hasn't been read yet -- leave it running.
            for step, process in list(running.items()):
                if not process.is_alive() and process.exitcode != 0:
                    running.pop(step).join()
                    progress.write(f"time step {step} CRASHED (exit code {process.exitcode})")
                    progress.update(1)
                    launch()
            continue

        running.pop(time_step).join()
        if error is not None:
            # tqdm.write, so the message does not collide with the progress bar
            progress.write(f"time step {time_step} FAILED: {error}")
        else:
            index = index_of[time_step]
            for name, values in zip(FIELDS, result):
                products[name][index] = values
            # tqdm.write, so each line scrolls above the bar instead of crowding it
            worst = "  ".join(
                f"{name.removeprefix('product_')} {products[name][index].min():.4f}"
                for name in FIELDS
            )
            progress.write(f"step {time_step:>3}: worst  {worst}")
        progress.update(1)
        launch()
    progress.close()

    done = ~np.isnan(products["product_raw"]).any(axis=1)
    print(f"\ncomputed {done.sum()}/{len(time_steps)} time steps")
    if done.any():
        print("scalar product over all computed time steps and modes:")
        for name in FIELDS:
            below = np.nansum(products[name] < V_MIN)
            print(f"  {name.removeprefix('product_'):>4}: "
                  f"worst {np.nanmin(products[name]):.4f}, "
                  f"median {np.nanmedian(products[name]):.4f}, "
                  f"{below}/{int(done.sum()) * N} below the colour floor of {V_MIN}")

    # Image per geometry: time step on x, mode index on y, scalar product as colour.
    # RdBu runs red (low) to blue (high), so poor agreement stands out red.
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5), sharey=True)
    extent = (time_steps[0] - 0.5, time_steps[-1] + 0.5, -0.5, N - 0.5)
    titles = (
        "WH vs FEM, raw (non-symmetric) geometry",
        "WH vs FEM, rotationally symmetric geometry",
        "FEM symmetric vs FEM raw (resampled)",
    )

    for ax, name, title in zip(axes, FIELDS, titles):
        im = ax.imshow(products[name].T, vmin=V_MIN, vmax=1.0, cmap="RdBu",
                       origin="lower", aspect="auto", extent=extent,
                       interpolation="nearest")
        ax.set_xlabel("time step")
        ax.set_title(title)

    axes[0].set_ylabel("FEM harmonic index")
    fig.colorbar(im, ax=axes, label=r"$\langle u_{FEM},\, w_{WH}\rangle$",
                 extend="min")
    fig.suptitle(
        f"Webster-Horn vs FEM eigenfunctions across development "
        f"({N} modes, mesh size {MESH_SIZE}; white gaps are missing time steps)"
    )

    output_path = os.path.join(RESULTS_DIR, "webster_horn_timeline.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {output_path}")
