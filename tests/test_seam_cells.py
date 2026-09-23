import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from embryoharmonics.celegans import SEAM_CELLS, SeamCellModelLoader
from embryoharmonics.celegans.gene_data import LOCATION_SCALE


@pytest.fixture(scope="module")
def parquet_dir(tmp_path_factory):
    """Two time steps of seam cells: at 400 all lie on the lateral midline
    (straightened frame); at 300 the T pair is missing and DV is nonzero.
    """
    names = [n for pair in SEAM_CELLS.values() for n in pair]
    ids = np.arange(1, len(names) + 1)
    z = np.repeat(np.linspace(30, 160, len(SEAM_CELLS)), 2)
    r = np.repeat(np.linspace(6, 2, len(SEAM_CELLS)), 2)
    lr = r * np.tile([1, -1], len(SEAM_CELLS))

    xyz = pa.table({
        "iLI": np.concatenate([ids, ids[:-2]]),
        "TI": np.concatenate([np.full(len(ids), 400), np.full(len(ids) - 2, 300)]),
        "LR": np.concatenate([lr, lr[:-2]]),
        "DV": np.concatenate([np.zeros(len(ids)), np.ones(len(ids) - 2)]),
        "AP": np.concatenate([z, z[:-2]]),
    })
    directory = tmp_path_factory.mktemp("parquet")
    pq.write_table(xyz, directory / "xyz.parquet")
    pq.write_table(pa.table({"i": ids, "LI": names}), directory / "xyz-lineages.parquet")
    return str(directory)


def test_only_complete_straightened_time_steps(parquet_dir):
    loader = SeamCellModelLoader(parquet_dir)
    assert loader.time_steps == [400]
    with pytest.raises(ValueError):
        loader.load(300)


def test_model_interpolates_seam_cells(parquet_dir):
    model = SeamCellModelLoader(parquet_dir).load(400)
    assert model.n_seam_cells == 11
    assert model.n_transverse_splines == 32

    t = model.spline_domain
    radius = np.linalg.norm(model.transverse_splines[0](t) - model.central_spline(t), axis=1)
    np.testing.assert_allclose(t[1:], np.linspace(30, 160, 10) * LOCATION_SCALE)
    np.testing.assert_allclose(radius[1:], np.linspace(6, 2, 10) * LOCATION_SCALE)
    assert t[0] == 0 and np.isclose(radius[0], radius[-1])


def test_model_generates_meridian_mesh(parquet_dir):
    mesh = SeamCellModelLoader(parquet_dir).load(400).generate_meridian_mesh(mesh_size=20)
    assert mesh.n_cells > 0
    assert mesh.bounds[5] == pytest.approx(160 * LOCATION_SCALE)
