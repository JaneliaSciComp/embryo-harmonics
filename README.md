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

