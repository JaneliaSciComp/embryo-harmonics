import logging
import os

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from embryoharmonics.celegans.embryo_model import EmbryoModel
from embryoharmonics.celegans.gene_data import LOCATION_SCALE


_logger = logging.getLogger(__name__)

# Embryonic lineage names of the seam cells (Sulston & White 1988, via the
# WormAtlas cell list), anterior to posterior; left and right of each pair.
SEAM_CELLS = {
    "H0": ("ABplaaappa", "ABarpapppa"),
    "H1": ("ABplaaappp", "ABarpapppp"),
    "H2": ("ABarppaaap", "ABarpppaap"),
    "V1": ("ABarppapaa", "ABarppppaa"),
    "V2": ("ABarppapap", "ABarppppap"),
    "V3": ("ABplappapa", "ABprappapa"),
    "V4": ("ABarppappa", "ABarpppppa"),
    "V5": ("ABplapapaap", "ABprapapaap"),
    "V6": ("ABarppappp", "ABarpppppp"),
    "T": ("ABplappppp", "ABprappppp"),
}

# The reference models replicate the 0-th transverse spline this many times
N_TRANSVERSE_SPLINES = 32


class SeamCellModelLoader:
    """Build rotationally symmetric embryo models from the seam cell positions
    in a directory of parquet files (``xyz.parquet``, ``xyz-lineages.parquet``).

    The seam cells are the lateral-most cells of the (straightened) embryo and
    come in left/right pairs, so their positions trace the worm outline: the
    radius profile is a natural cubic spline through (AP, |LR|) of the pairs,
    closed anteriorly at AP = 0 with the radius of the T pair. Time steps are
    those at which all seam cells exist and lie on the lateral midline, i.e.
    the embryo is in the straightened frame (from V5's birth onwards).
    """
    def __init__(self, directory: str):
        self.directory = directory
        lineages = pq.read_table(os.path.join(directory, "xyz-lineages.parquet")).to_pydict()
        name_to_id = dict(zip(lineages["LI"], lineages["i"]))
        ids = np.array([[name_to_id[n] for n in pair] for pair in SEAM_CELLS.values()])

        xyz = pq.read_table(os.path.join(directory, "xyz.parquet"))
        xyz = xyz.filter(pc.is_in(xyz["iLI"], value_set=pa.array(ids.ravel())))
        time = xyz["TI"].to_numpy()
        cell = xyz["iLI"].to_numpy()
        self._xyz = np.stack([xyz[c].to_numpy() for c in ("LR", "DV", "AP")], axis=1)

        # Per time step, the (pair, side) -> row index table of the seam cells
        self._rows = {}
        for t in np.unique(time):
            at_t = np.flatnonzero(time == t)
            rows = np.full(ids.shape, -1)
            for r in at_t:
                rows[ids == cell[r]] = r
            if (rows >= 0).all() and np.allclose(self._xyz[rows, 1], 0):
                self._rows[int(t)] = rows


    @property
    def time_steps(self) -> list[int]:
        return list(self._rows)


    def load(self, time_step: int, *, symmetric: bool = True) -> EmbryoModel:
        """Build the embryo model for the given time step (always symmetric;
        the parameter mirrors :meth:`EmbryoModelLoader.load`).

        :raises ValueError: If the time step has no complete straightened seam
        """
        if time_step not in self._rows:
            raise ValueError(f"Time step {time_step} has no complete seam cell outline in '{self.directory}'")

        _logger.info("Building seam cell model for time step %d from '%s'", time_step, self.directory)
        rows = self._rows[time_step]
        z = self._xyz[rows, 2].mean(axis=1)
        radius = np.abs(self._xyz[rows, 0]).mean(axis=1)
        # ponytail: nose at AP = 0 with the tail (T) radius; the reference h5
        # models use a similar anterior anchor (a0), swap in a real rule if found
        z = np.concatenate(([0.0], z)) * LOCATION_SCALE
        radius = np.concatenate(([radius[-1]], radius)) * LOCATION_SCALE

        central = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
        transverse = np.column_stack([radius, np.zeros_like(z), z])
        return EmbryoModel(
            ["a0", *SEAM_CELLS],
            z,
            central,
            [transverse] * N_TRANSVERSE_SPLINES,
            symmetric=True,
            bc_type="natural",
        )
