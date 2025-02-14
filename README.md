# ECM simulations
A collection of scripts for computing eigenfunctions on geometrical models of C elegans embryos using [NGSolve](https://ngsolve.org/).

### Setup
Run the following line from the terminal to create a new environment called `embryo-harmonics`:
```bash
conda env create -f environment.yaml
```

Activate the environment:
```bash
conda activate embryo-harmonics
```

Then, you should be able to run the examples using:
```bash
python <filename>  # just console output, no GUI
netgen <filename>  # console output and GUI
```

The `.py` scripts in the `scripts/` directory are text representations of notebook files courtesy of jupytext.
When you open them in jupyter (right-click and choose 'Open With > Jupytext notebook'), they will be automatically converted to interactive notebooks, which you can find in the `notebooks/` directory.
Saving a notebook will also automatically update the associated `.py` file.

