import pytest

import numpy as np

from embryoharmonics.celegans import GeneData


def test_construction_with_arbitrary_data():
    """Test the construction of a GeneData object with arbitrary data."""
    locations = np.array([[0, 0, 0], [1, 1, 1]])
    activities = np.array([0.5, 0.8])
    gene_data = GeneData("test_gene", locations, activities)

    assert len(gene_data) == 2


def test_construction_with_incompatible_data_raises_error():
    """Test that the GeneData constructor raises an error when locations and
    activities are of different length.
    """
    locations = np.array([[0, 0, 0], [1, 1, 1]])
    activities = np.array([0.5])

    with pytest.raises(ValueError):
        GeneData("test_gene", locations, activities)


def test_gene_data_as_point_cloud():
    """Test that the resulting point cloud has the correct data."""
    locations = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    activities = np.array([0.5, 0.8])
    gene_data = GeneData("test_gene", locations, activities)

    point_cloud = gene_data.as_point_cloud()

    assert point_cloud.n_points == 2
    assert all(point_cloud["test_gene"] == activities)


def test_gene_data_interpolation_conserves_mass(embryo_mesh, fem_matrices):
    """Test that the interpolation of gene data onto a mesh conserves the
    overall gene activity count.
    """
    locations = np.array([[0, 0, 1], [1, 1, 20]])
    activities = np.array([0.123, 0.321])

    gene_data = GeneData("test_gene", locations, activities)

    interpolated_mesh_data = gene_data.interpolate(embryo_mesh)
    total_mass = (interpolated_mesh_data.data @ fem_matrices.mass).sum()

    assert gene_data.activities.sum() == pytest.approx(total_mass, rel=1e-4)


def test_gene_data_loader_gene_names(gene_data_loader):
    """Test that the gene data loader returns the correct shape for gene names."""
    gene_names = gene_data_loader.gene_names

    assert isinstance(gene_names[0], str)
    assert len(gene_names) == gene_data_loader.n_genes


def test_gene_data_loader_tissue_names(gene_data_loader):
    """Test that the gene data loader returns the correct shape for tissue names."""
    tissue_names = gene_data_loader.tissue_names

    assert isinstance(tissue_names[0], str)
    assert len(tissue_names) == gene_data_loader.n_tissues


def test_gene_data_loader_time_steps(gene_data_loader):
    """Test that the gene data loader returns the correct shape for time steps."""
    time_steps = gene_data_loader.time_steps

    assert len(time_steps) == gene_data_loader.n_time_steps


def test_gene_data_loader_load(gene_data_loader):
    """Test that the gene data loader returns the correct gene."""
    gene_data = gene_data_loader.load("cwn-1", 420, remove_nans=False)

    assert gene_data.name == "cwn-1"
    assert len(gene_data) == gene_data_loader.n_cells


def test_removing_nans_works(gene_data_loader):
    """Test that removing nans from the gene data works."""
    gene_data_with_nans = gene_data_loader.load("cwn-1", 420, remove_nans=False)
    gene_data = gene_data_loader.load("cwn-1", 420, remove_nans=True)

    assert np.isnan(gene_data_with_nans.activities).sum() > 0
    assert np.isnan(gene_data.activities).sum() == 0
    assert len(gene_data) < len(gene_data_with_nans)


def test_gene_data_loader_load_tissue(gene_data_loader):
    """Test that the gene data loader returns the correct tissue."""
    tissue_data = gene_data_loader.load_tissue("muscle", 420)

    assert tissue_data.name == "muscle"
    assert len(tissue_data) == gene_data_loader.n_cells
