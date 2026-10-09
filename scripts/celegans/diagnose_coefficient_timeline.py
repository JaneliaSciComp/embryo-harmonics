"""Diagnose all-zero time points in a harmonic coefficient file: compare the
time lines and per-time-step coverage of the parquet gene data (expression
table vs. cell positions) with the computed coefficients, and write the plots
plus a markdown report into one directory.

Example:
    uv run python scripts/celegans/diagnose_coefficient_timeline.py \\
        data/cpm-20260917-all010 results/harmonic_coefficients_smoothed_2D_20261002.h5 \\
        results/embryo-harmonics_smoothed-2D_20261002.h5 --time-offset 380
"""
import argparse
import os

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

from embryoharmonics import celegans
from embryoharmonics.celegans import load_axisymmetric_harmonics

matplotlib.use("Agg")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("gene_dir", help="Directory of parquet gene data")
    parser.add_argument("coefficient_file", help="HDF5 file with harmonic coefficients")
    parser.add_argument("geometry_file", help="HDF5 file with the (axisymmetric) meshes and harmonics")
    parser.add_argument("--time-offset", type=int, default=380, help="As passed to process_all_genes (default: %(default)s)")
    parser.add_argument("--buffer", type=float, nargs="+", default=[0.0], help="As passed to process_all_genes (default: %(default)s)")
    parser.add_argument("--n-genes", type=int, default=6, help="Number of example gene trajectories (default: %(default)s)")
    parser.add_argument("--out-dir", default="results/coefficient_timeline_report")
    return parser.parse_args()


def aggregate_gene_data(gene_dir, cache):
    """Per time point of the expression table: CPM sums and non-zero counts per
    gene over all lineages and over lineages that have a cell position at that
    time; lineage counts; (time, lineage, has position) list. Cached as npz
    because the expression table has hundreds of millions of rows.
    """
    if os.path.exists(cache):
        return dict(np.load(cache, allow_pickle=True))

    path = lambda n: os.path.join(gene_dir, n)
    genes = pq.read_table(path("genes.parquet")).to_pydict()
    cpm_lin = pq.read_table(path("lineages.parquet")).to_pydict()
    xyz_lin = pq.read_table(path("xyz-lineages.parquet")).to_pydict()
    xyz = pq.read_table(path("xyz.parquet"), columns=["iLI", "TI"])
    cpm = pq.read_table(path("cpm.parquet"), columns=["iGE", "iLI", "TI", "CPM_mean"])
    ti, li, ge, val = (cpm[c].to_numpy() for c in ("TI", "iLI", "iGE", "CPM_mean"))

    times = np.unique(ti)
    ti_idx = np.searchsorted(times, ti).astype(np.int64)
    n_ge = max(genes["i"]) + 1
    n_li = max(cpm_lin["i"]) + 1

    # Which expression lineages have a position at which time (matched by name)
    xyz_name_to_cpm_id = dict(zip(cpm_lin["LI"], cpm_lin["i"]))
    xyz_id_to_cpm_id = np.full(max(xyz_lin["i"]) + 1, -1)
    xyz_id_to_cpm_id[xyz_lin["i"]] = [xyz_name_to_cpm_id.get(n, -1) for n in xyz_lin["LI"]]
    xyz_ti, xyz_li = xyz["TI"].to_numpy(), xyz["iLI"].to_numpy()
    has_pos = np.zeros((len(times), n_li), dtype=bool)
    in_window = np.isin(xyz_ti, times)
    cpm_ids = xyz_id_to_cpm_id[xyz_li[in_window]]
    ok = cpm_ids >= 0
    has_pos[np.searchsorted(times, xyz_ti[in_window][ok]), cpm_ids[ok]] = True
    positioned = has_pos[ti_idx, li]

    key = ti_idx * n_ge + ge
    shape = (len(times), n_ge)
    def table(mask, weights):
        return np.bincount(key[mask], weights=weights[mask], minlength=np.prod(shape)).reshape(shape)
    everything = np.ones(len(ti), dtype=bool)
    nonzero = val > 0
    sum_all = table(everything, val)
    sum_pos = table(positioned, val)
    nnz_all = table(everything, nonzero.astype(float))
    nnz_pos = table(positioned, nonzero.astype(float))

    # (time, lineage) pairs present in the expression table
    pairs = np.unique(ti_idx * n_li + li)
    pair_ti, pair_li = pairs // n_li, pairs % n_li
    pair_has_pos = has_pos[pair_ti, pair_li]
    cpm_lin_names = np.array([""] * n_li, dtype=object)
    cpm_lin_names[cpm_lin["i"]] = cpm_lin["LI"]
    gene_names = np.array([""] * n_ge, dtype=object)
    gene_names[genes["i"]] = genes["GE"]

    xyz_times, xyz_counts = np.unique(xyz_ti, return_counts=True)
    xyz_names = np.array([""] * (max(xyz_lin["i"]) + 1), dtype=object)
    xyz_names[xyz_lin["i"]] = xyz_lin["LI"]

    result = dict(
        times=times, gene_names=gene_names, gene_ids=np.array(genes["i"]),
        sum_all=sum_all, sum_pos=sum_pos, nnz_all=nnz_all, nnz_pos=nnz_pos,
        pair_time=times[pair_ti], pair_lineage=cpm_lin_names[pair_li], pair_has_pos=pair_has_pos,
        xyz_times=xyz_times, xyz_counts=xyz_counts,
        xyz_pair_time=xyz_ti, xyz_pair_lineage=xyz_names[xyz_li],
    )
    np.savez(cache, **result)
    return result


def load_coefficients(file_name):
    with h5py.File(file_name, "r") as f:
        coeff = f["gene_coefficients"][:]
        times = f["time_points"][:]
    return times, coeff


def inside_body_per_time(gene_dir, geometry_file, times, time_offset, buffer):
    """Per coefficient time point: number of cells, cells inside the body
    (harmonics sample to non-zero there) and cells that carry any expression."""
    gdl = celegans.open_gene_data_loader(
        gene_dir, location_scale=celegans.gene_data.buffered_location_scale(buffer))
    rows = []
    for t in times:
        locations, activities = gdl.load_all(int(t))
        samples = load_axisymmetric_harmonics(geometry_file, int(t) - time_offset).sample(locations)
        inside = np.abs(samples).sum(0) > 0
        expressed = ~np.isnan(activities).all(0)
        rows.append((len(locations), inside.sum(), expressed.sum(), (inside & expressed).sum()))
    return np.array(rows)


def mark_zero_times(ax, zero_times):
    for t in zero_times:
        ax.axvline(t, color="red", lw=0.8, alpha=0.6)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    out = lambda n: os.path.join(args.out_dir, n)

    agg = aggregate_gene_data(args.gene_dir, out("gene_data_aggregates.npz"))
    gd_times = agg["times"]
    coeff_times, coeff = load_coefficients(args.coefficient_file)
    ids = agg["gene_ids"]
    assert coeff.shape[1] == len(ids), "gene count mismatch between parquet and coefficient file"
    coeff_nnz_genes = (np.abs(coeff).sum(2) > 0).sum(1)
    zero_times = coeff_times[coeff_nnz_genes == 0]
    sel = np.searchsorted(gd_times, coeff_times)  # rows of the aggregates for coefficient times
    model_times = np.arange(1, 372) + args.time_offset

    # Per time point of the expression table
    n_lin = np.array([(agg["pair_time"] == t).sum() for t in gd_times])
    n_lin_pos = np.array([((agg["pair_time"] == t) & agg["pair_has_pos"]).sum() for t in gd_times])
    genes_nz_all = (agg["nnz_all"][:, ids] > 0).sum(1)
    genes_nz_pos = (agg["nnz_pos"][:, ids] > 0).sum(1)

    # --- Figure 1: non-zero genes per time step, data vs. coefficients
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.axvspan(model_times[0], model_times[-1], color="0.92", label="geometry models available")
    ax.plot(gd_times, genes_nz_all, "o-", ms=3, label="expression table: genes with CPM > 0 (any lineage)")
    ax.plot(gd_times, genes_nz_pos, "s-", ms=3, label="expression table: genes with CPM > 0 in lineages that have a cell position")
    ax.plot(coeff_times, coeff_nnz_genes, "x--", ms=6, color="k", label="coefficient file: genes with any non-zero coefficient")
    mark_zero_times(ax, zero_times)
    ax.set(xlabel="time point TI (min post first cleavage)", ylabel="number of genes", title="Non-zero genes per time step")
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout(); fig.savefig(out("01_nonzero_genes_per_time.png"), dpi=130); plt.close(fig)

    # --- Figure 2: time lines
    fig, ax = plt.subplots(figsize=(12, 3.2))
    rows = [("positions (xyz.parquet)", agg["xyz_times"]), ("expression (cpm.parquet)", gd_times),
            ("geometry models", model_times), ("coefficients", coeff_times), ("all-zero coefficients", zero_times)]
    ax.eventplot([r[1] for r in rows], lineoffsets=range(len(rows)), linelengths=0.8,
                 colors=["C0", "C1", "0.5", "k", "red"])
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    ax.set(xlabel="time point TI", title="Which time points exist where (coefficients = expression ∩ positions ∩ models)")
    fig.tight_layout(); fig.savefig(out("02_timelines.png"), dpi=130); plt.close(fig)

    # --- Figure 3: lineages per time step and how many have a position
    xyz_count_at = dict(zip(agg["xyz_times"], agg["xyz_counts"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(gd_times, n_lin, "o-", ms=3, label="lineages with expression")
    axes[0].plot(gd_times, n_lin_pos, "s-", ms=3, label="... of which have a cell position at that time")
    axes[0].plot(gd_times, [xyz_count_at.get(t, 0) for t in gd_times], "-", color="0.5", label="cells with a position")
    axes[0].set(ylabel="count", title="Lineage coverage per expression time point"); axes[0].legend(fontsize=8)
    frac = np.where(n_lin > 0, 1 - n_lin_pos / np.maximum(n_lin, 1), np.nan)
    axes[1].plot(gd_times, 100 * frac, "o-", ms=3, color="C3")
    axes[1].set(ylabel="% of expression dropped\n(no cell position)", xlabel="time point TI", ylim=(-2, 102))
    for ax in axes:
        mark_zero_times(ax, zero_times); ax.axvspan(model_times[0], model_times[-1], color="0.92")
    fig.tight_layout(); fig.savefig(out("03_lineage_coverage.png"), dpi=130); plt.close(fig)

    # --- Figure 4: lineage x time raster inside the coefficient window
    in_win = np.isin(agg["pair_time"], coeff_times)
    lineages = np.array(sorted(set(agg["pair_lineage"][in_win])))
    raster = np.zeros((len(lineages), len(coeff_times)))
    li_idx = {l: i for i, l in enumerate(lineages)}
    for t, l, p in zip(agg["pair_time"][in_win], agg["pair_lineage"][in_win], agg["pair_has_pos"][in_win]):
        raster[li_idx[l], np.searchsorted(coeff_times, t)] = 2 if p else 1
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.imshow(raster, aspect="auto", interpolation="nearest", cmap=matplotlib.colors.ListedColormap(["white", "C3", "C0"]), vmin=0, vmax=2)
    ax.set_xticks(range(len(coeff_times))); ax.set_xticklabels(coeff_times, rotation=90, fontsize=7)
    for i, t in enumerate(coeff_times):
        if t in zero_times:
            ax.get_xticklabels()[i].set_color("red")
    step = max(1, len(lineages) // 60)
    ax.set_yticks(range(0, len(lineages), step)); ax.set_yticklabels(lineages[::step], fontsize=6)
    ax.set(xlabel="time point TI (red = all-zero coefficients)", ylabel="lineage with expression data",
           title="Expression table within the coefficient window: blue = lineage has a cell position, red = no position (dropped)")
    fig.tight_layout(); fig.savefig(out("04_lineage_time_raster.png"), dpi=130); plt.close(fig)

    # --- Figure 5: cells inside the body / with expression per coefficient time point
    body = inside_body_per_time(args.gene_dir, args.geometry_file, coeff_times, args.time_offset, args.buffer)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(coeff_times, body[:, 0], "-", color="0.5", label="cells with position")
    ax.plot(coeff_times, body[:, 1], "o-", ms=3, label="... inside the body (harmonics non-zero)")
    ax.plot(coeff_times, body[:, 2], "s-", ms=3, label="... with any expression value")
    ax.plot(coeff_times, body[:, 3], "x--", ms=5, color="k", label="... inside the body and with expression (contribute to coefficients)")
    mark_zero_times(ax, zero_times)
    ax.set(xlabel="time point TI", ylabel="number of cells", title="Cells that can contribute to the coefficients")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(out("05_cells_inside_body.png"), dpi=130); plt.close(fig)

    # --- Figure 6: example gene trajectories
    sum_pos_win = agg["sum_pos"][sel][:, ids]
    top = np.argsort(sum_pos_win.sum(0))[::-1][: args.n_genes]
    coeff_norm = np.linalg.norm(coeff, axis=2)
    fig, axes = plt.subplots(len(top), 1, figsize=(12, 2.3 * len(top)), sharex=True)
    for ax, g in zip(np.atleast_1d(axes), top):
        name = agg["gene_names"][ids[g]]
        def norm(y): return y / y.max() if y.max() > 0 else y
        ax.plot(gd_times, norm(agg["sum_all"][:, ids[g]]), "o-", ms=3, label="Σ CPM over all lineages")
        ax.plot(gd_times, norm(agg["sum_pos"][:, ids[g]]), "s-", ms=3, label="Σ CPM over lineages with a position")
        ax.plot(coeff_times, norm(np.abs(coeff[:, g, 0])), "^--", ms=4, color="k", label="|coefficient of mode 0|")
        ax.plot(coeff_times, norm(coeff_norm[:, g]), "x:", ms=4, color="0.4", label="‖all coefficients‖")
        mark_zero_times(ax, zero_times); ax.axvspan(model_times[0], model_times[-1], color="0.92")
        ax.set_ylabel(name, fontsize=9); ax.set_ylim(-0.05, 1.05)
    np.atleast_1d(axes)[0].legend(fontsize=7, ncol=4, loc="upper left")
    np.atleast_1d(axes)[0].set_title("Example gene trajectories (each series scaled to its maximum)")
    np.atleast_1d(axes)[-1].set_xlabel("time point TI")
    fig.tight_layout(); fig.savefig(out("06_gene_trajectories.png"), dpi=130); plt.close(fig)

    # --- Figure 7: mode-0 coefficient vs. positioned CPM sum, all genes and times
    fig, ax = plt.subplots(figsize=(6, 5.5))
    x, y = sum_pos_win.ravel(), np.abs(coeff[:, :, 0]).ravel()
    keep = (x > 0) | (y > 0)
    ax.loglog(x[keep] + 1e-3, y[keep] + 1e-6, ".", ms=1, alpha=0.3)
    ax.set(xlabel="Σ CPM over lineages with a position (+1e-3)", ylabel="|coefficient of mode 0| (+1e-6)",
           title="Coefficients vs. positioned expression\n(all genes × coefficient time points)")
    fig.tight_layout(); fig.savefig(out("07_coefficient_vs_positioned_cpm.png"), dpi=130); plt.close(fig)

    # --- Markdown report
    xyz_pairs = set(zip(agg["xyz_pair_time"].tolist(), agg["xyz_pair_lineage"].tolist()))
    post_start = args.time_offset
    post_cells = {l for (t, l) in xyz_pairs if t > post_start}
    ref = int(np.argmax(n_lin[sel]))  # best covered coefficient time point as reference
    def last_position_time(lineage):
        ts = [t for (t, l) in xyz_pairs if l == lineage]
        return f"{min(ts)}–{max(ts)}" if ts else "never"
    lines = [
        "# Where do the all-zero time points in the coefficients come from?", "",
        f"Coefficients: `{args.coefficient_file}`  ", f"Gene data: `{args.gene_dir}`  ",
        f"Geometry: `{args.geometry_file}` (time offset {args.time_offset}, buffer {args.buffer})", "",
        "## Summary", "",
        f"- The expression table has {len(gd_times)} distinct time points, the positions {len(agg['xyz_times'])}, "
        f"the geometry models {len(model_times)} ({model_times[0]}–{model_times[-1]}). "
        f"Coefficients exist at their intersection: {len(coeff_times)} time points ({coeff_times[0]}–{coeff_times[-1]}).",
        f"- {len(zero_times)} coefficient time points are all zero: {', '.join(map(str, zero_times))}.",
        "- Expression is not sampled on a regular grid: each time point of the expression table carries only the lineages "
        "that were assigned that time, from 1 to several hundred (Fig. 3, 4). The coefficient pipeline joins expression to "
        "cell positions by exact lineage name at the same time point and drops everything else (`ParquetGeneDataLoader`).",
        "- After twitching (TI > {0}) the position table contains a fixed set of {1} named cells; expression lineages outside this set "
        "(e.g. {2} of {3} lineages at TI {4}) exist in the position table only before twitching and are dropped entirely.".format(
            post_start, len(post_cells), n_lin[sel][ref] - n_lin_pos[sel][ref], n_lin[sel][ref], coeff_times[ref]),
        "- At the all-zero time points the expression table contains only 1–3 lineages, and none of them belongs to the post-twitch "
        "position set (table below). All expression at that time is dropped, so every coefficient is zero. The same mechanism explains "
        "the strongly fluctuating number of non-zero genes at the other time points (Fig. 1, 3).",
        "- Where coefficients are non-zero they track the CPM sum over positioned lineages (Fig. 6, 7): apart from the lineage join, "
        "only the handful of cells that lie outside the body are lost (bottom row in Fig. 7).",
        "- Geometry is not the cause: the number of cells inside the body is nearly constant over time (Fig. 5).", "",
        "## All-zero time points", "",
        "| TI | lineages with expression | position at TI | in post-twitch position set | TI range with a position |",
        "|---|---|---|---|---|",
    ]
    for t in zero_times:
        m = agg["pair_time"] == t
        for l, p in zip(agg["pair_lineage"][m], agg["pair_has_pos"][m]):
            lines.append(f"| {t} | {l} | {'yes' if p else 'no'} | {'yes' if l in post_cells else 'no'} | {last_position_time(l)} |")
    lines += ["", "## Per coefficient time point", "",
              "| TI | lineages w/ expression | w/ position | genes CPM>0 (all) | genes CPM>0 (positioned) | genes non-zero coeff. | cells inside body | cells contributing |",
              "|---|---|---|---|---|---|---|---|"]
    for i, t in enumerate(coeff_times):
        j = sel[i]
        lines.append(f"| {t} | {n_lin[j]} | {n_lin_pos[j]} | {genes_nz_all[j]} | {genes_nz_pos[j]} | {coeff_nnz_genes[i]} | {body[i, 1]} | {body[i, 3]} |")
    lines += ["", "## Figures", ""] + [
        f"### {title}\n\n![]({name})\n" for name, title in [
            ("01_nonzero_genes_per_time.png", "Non-zero genes per time step"),
            ("02_timelines.png", "Time lines of the inputs"),
            ("03_lineage_coverage.png", "Lineage coverage per expression time point"),
            ("04_lineage_time_raster.png", "Lineage × time raster in the coefficient window"),
            ("05_cells_inside_body.png", "Cells inside the body and with expression"),
            ("06_gene_trajectories.png", "Example gene trajectories"),
            ("07_coefficient_vs_positioned_cpm.png", "Mode-0 coefficient vs. positioned CPM sum"),
        ]]
    with open(out("report.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Report written to {out('report.md')}; all-zero time points: {zero_times.tolist()}")


if __name__ == "__main__":
    main()
