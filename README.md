# ECM simulations
A collection of scripts for computing eigenfunctions on geometrical models of C elegans embryos using [NGSolve](https://ngsolve.org/).

### Setup
Install [uv](https://docs.astral.sh/uv/), then create the environment:
```bash
uv sync
```

Then, you should be able to run the examples using:
```bash
uv run python <filename>  # just console output, no GUI
uv run netgen <filename>  # console output and GUI
```

The `.py` scripts in the `scripts/` directory are text representations of notebook files courtesy of jupytext.
When you open them in jupyter (right-click and choose 'Open With > Jupytext notebook'), they will be automatically converted to interactive notebooks, which you can find in the `notebooks/` directory.
Saving a notebook will also automatically update the associated `.py` file.

### C. elegans CLI scripts
`scripts/celegans/process_all_geometries.py` and `scripts/celegans/process_all_genes.py` are plain CLI scripts (not jupytext notebooks). Run `-h` on either for the full list of options.

Generate meshes and harmonics for every time step of an embryo model.
They are stored in a single HDF5 file with a sibling `.xdmf` file that can be opened in ParaView:
```bash
uv run python scripts/celegans/process_all_geometries.py <path_to_celegans_models.h5> \
    --output-file results/embryo_harmonics.h5 --n-harmonics 300 --mesh-size 5
```

Compute harmonic coefficients for all genes and tissues from the meshes/harmonics produced above:
```bash
uv run python scripts/celegans/process_all_genes.py <path_to_celegans_genedata.h5> results/embryo_harmonics.h5 \
    --output-file harmonic_coefficients.h5
```


### Pre-twitch (ovoid) CLI scripts
Before twitching, the embryo is an ovoid whose outline is a Fourier series of the radius around its centroid (`data/ovoid`).
`scripts/ovoid-symmetric` computes the exact harmonics of the body of revolution, `scripts/ovoid-horn-z` and `scripts/ovoid-horn-phi` Webster-Horn approximations in the axial coordinate and in the polar angle around the centroid (see `scripts/ovoid-horn-phi/horn_z_vs_phi.html` for the derivations and a comparison).
`scripts/ovoid-sphere-map` stretches the exact ball harmonics radially onto the ovoid, without solving anything.
There is one geometry for all pre-twitch time points:
```bash
uv run python scripts/ovoid-symmetric/process_geometry.py data/ovoid/ovoid_coefficient_comparison.csv \
    --output-file results/exact_harmonics.h5
uv run python scripts/ovoid-horn-z/process_geometry.py data/ovoid/ovoid_coefficient_comparison.csv \
    --output-file results/horn_z_natural_harmonics.h5 --caps natural  # or flat, corrected
```
The outline is in pixels of 0.0391 microns. By default, it is scaled up by 1% (`--radial-extension`) so that all cells fit, and its first tip points to the negative AP direction (`--no-flip` reverts that).
The gene script (a copy in every folder) fits one translation that moves the cells of all time points up to `--last-time-step` into the ovoid and stores it with the coefficients:
```bash
uv run python scripts/ovoid-symmetric/process_all_genes.py data/embryo-20261009.zarr results/exact_harmonics.h5 \
    --output-file exact_coefficients.h5
```
`scripts/ovoid-horn-phi/compare_approximations.py results` compares all `<variant>_harmonics.h5` in a directory with `exact_harmonics.h5`.
Afterwards, `scripts/ovoid-horn-phi/make_report.py results` writes a short HTML report with the metrics and the best-matching modes of the z-horn, the φ-horn and the sphere map.
