import argparse
import base64
import csv
import io
import os

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pyvista as pv

from compare_approximations import EXACT, fields_of
from embryoharmonics.celegans import load_axisymmetric_harmonics
from embryoharmonics.fem import compute_mass_and_stiffness_axisymmetric


# The approximations whose modes are shown, with their display names
SHOWN = {
    "horn_z_natural": "z-horn (natural caps)",
    "horn_phi": "φ-horn",
    "sphere_map": "sphere map",
}
NAMES = {
    EXACT: "exact (2D meridian FEM)",
    "horn_z_natural": "z-horn, natural caps",
    "horn_z_flat": "z-horn, flat caps",
    "horn_z_corrected": "z-horn, corrected",
    "horn_phi": "φ-horn",
    "sphere_map": "sphere map",
}
# (column in approximation_metrics.csv, header, whether it is a percentage)
COLUMNS = [
    ("mean_eigenvalue_error_first_30", "eigenvalue error<br>first 30", True),
    ("mean_eigenvalue_error_all", "eigenvalue error<br>all", True),
    ("mean_best_match_first_30", "best match<br>first 30", False),
    ("mean_best_match_all", "best match<br>all", False),
    ("subspace_capture_30", "subspace capture<br>30", False),
    ("subspace_capture_300", "subspace capture<br>300", False),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Write a short HTML report comparing the approximate ovoid "
        "harmonics with the exact ones: a table of the metrics from "
        "compare_approximations.py (run it first) and the best-matching "
        "approximate modes of a few low, middle and high exact modes."
    )
    parser.add_argument("directory", help="Directory with the results of all variants")
    parser.add_argument(
        "--output-file",
        default=None,
        help="HTML file to write (default: <directory>/approximation_report.html)",
    )
    return parser.parse_args()


def pick_modes(exact):
    """A few distinct (cos) exact modes: the four lowest non-constant ones,
    and one axisymmetric and one non-axisymmetric mode from the middle and
    the end of the spectrum.
    """
    cos = np.flatnonzero(exact.trig_kinds == "cos")
    axisymmetric = exact.angular_orders[cos] == 0
    middle = cos >= len(exact) // 2
    low = list(cos[cos > 0][:4])
    middle = [cos[middle & axisymmetric][0], cos[middle & ~axisymmetric][0]]
    high = sorted([cos[axisymmetric][-1], cos[~axisymmetric][-1]])
    return low, middle, high


def to_png(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode()


def mode_figure(exact, approximations, rows, mass):
    """One row per exact mode, one column per variant: the exact meridian
    section and the best-matching approximate mode of the same angular
    order, in the plane theta = 0 (top) / pi (bottom).
    """
    points = exact.mesh.points
    triangles = exact.mesh.cells_dict[pv.CellType.TRIANGLE]
    mirrored = mtri.Triangulation(
        np.concatenate([points[:, 2], points[:, 2]]),
        np.concatenate([points[:, 0], -points[:, 0]]),
        np.vstack([triangles, triangles + len(points)]),
    )

    def normalized(harmonics):
        fields = fields_of(harmonics)
        return fields / np.sqrt(np.einsum("ij,ij->i", fields, (mass @ fields.T).T))[:, None]

    exact_fields = normalized(exact)
    approximate_fields = {name: normalized(h) for name, h in approximations.items()}

    columns = {EXACT: NAMES[EXACT], **SHOWN}
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(2.6 * len(columns), 1.55 * len(rows)))
    for row, j in enumerate(rows):
        m = exact.angular_orders[j]
        for col, name in enumerate(columns):
            ax = axes[row, col]
            if name == EXACT:
                field, title = exact_fields[j], f"#{j}  m={m}  λ={exact.eigenvalues[j]:.3g}"
            else:
                harmonics = approximations[name]
                products = approximate_fields[name] @ (mass @ exact_fields[j])
                same = (harmonics.angular_orders == m) & (harmonics.trig_kinds == "cos")
                i = np.argmax(np.where(same, np.abs(products), -1))
                field = np.sign(products[i]) * approximate_fields[name][i]
                error = harmonics.eigenvalues[i] / exact.eigenvalues[j] - 1
                title = f"#{i}  match {abs(products[i]):.2f}  Δλ {100 * error:+.1f}%"
            values = np.concatenate([field, (-1) ** m * field])
            limit = np.abs(values).max()
            ax.tripcolor(mirrored, values, cmap="RdBu_r", vmin=-limit, vmax=limit,
                         shading="gouraud")
            ax.set_aspect("equal")
            ax.set_axis_off()
            ax.set_title(title, fontsize=7)
            if row == 0:
                ax.text(0.5, 1.35, columns[name], transform=ax.transAxes,
                        ha="center", fontsize=9, weight="bold")
    fig.tight_layout(h_pad=0.6)
    return to_png(fig)


def metrics_table(directory):
    with open(os.path.join(directory, "approximation_metrics.csv")) as file:
        rows = {row["variant"]: row for row in csv.DictReader(file)}
    order = [v for v in NAMES if v in rows and v != EXACT]
    header = "".join(f"<th>{title}</th>" for _, title, _ in COLUMNS)
    body = []
    for variant in order:
        cells = []
        for column, _, percent in COLUMNS:
            value = float(rows[variant][column]) if rows[variant].get(column) else np.nan
            cells.append(f"<td>{100 * value:.1f}%</td>" if percent else f"<td>{value:.3f}</td>")
        body.append(f"<tr><th>{NAMES[variant]}</th>{''.join(cells)}</tr>")
    return f"<table><thead><tr><th>approximation</th>{header}</tr></thead>" \
           f"<tbody>{''.join(body)}</tbody></table>"


TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ovoid Harmonics Comparison</title>
<style>
:root {{ --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --line: #d9dce1; --head: #f3f4f6; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
  --bg: #16181c; --fg: #e8eaed; --muted: #a0a4aa; --line: #343841; --head: #20232a; }} }}
:root[data-theme="dark"] {{ --bg: #16181c; --fg: #e8eaed; --muted: #a0a4aa; --line: #343841; --head: #20232a; }}
body {{ background: var(--bg); color: var(--fg); font: 15px/1.5 system-ui, sans-serif;
  max-width: 1080px; margin: 0 auto; padding: 24px 16px 48px; }}
h1 {{ font-size: 1.5rem; margin-bottom: 0.2rem; }}
h2 {{ font-size: 1.15rem; margin-top: 2rem; }}
p, li {{ color: var(--fg); }} .muted {{ color: var(--muted); font-size: 0.9rem; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 640px; }}
th, td {{ border-bottom: 1px solid var(--line); padding: 6px 10px; text-align: right; }}
thead th {{ background: var(--head); font-weight: 600; font-size: 0.85rem; }}
tbody th {{ text-align: left; font-weight: 500; }}
img {{ max-width: 100%; height: auto; background: #fff; border-radius: 4px; }}
</style></head><body>
<h1>Harmonics on the ovoid embryo</h1>
<p class="muted">{n_modes} Neumann harmonics of the pre-twitch ovoid ({n_points} meridian nodes),
approximated in three ways and compared with the exact ones.</p>
<ul>
<li><b>exact</b>: finite elements on the 2D meridian section, one problem per angular order m.</li>
<li><b>z-horn</b>: Webster-horn equation in the radius R(z) along the axis (three cap treatments).</li>
<li><b>φ-horn</b>: horn equation in the polar outline r(φ) around the centroid, with the
radial shape of a ball of the local radius and an envelope A(φ) from a 1D problem.</li>
<li><b>sphere map</b>: the exact ball harmonics j<sub>k</sub>(κρ) P<sub>k</sub><sup>m</sup>(cos φ),
stretched radially onto the ovoid (ρ → ρ / r(φ)); eigenvalues are their Rayleigh quotients.
Nothing is solved.</li>
</ul>

<h2>Errors</h2>
<div class="scroll">{table}</div>
<p class="muted">Eigenvalue error: mean |λ<sub>i</sub> / λ<sub>i</sub><sup>exact</sup> − 1| over
modes in order. Best match: for every exact mode, the largest |scalar product| with a normalized
approximate mode of the same angular order (1 = identical shape). Subspace capture: mean squared
projection of the first N exact modes onto the span of the first N approximate ones.</p>

<h2>Modes</h2>
<p class="muted">Rows: exact modes from the low end ({low}), the middle ({middle}) and the end
({high}) of the spectrum. Columns: the best-matching approximate mode with its index, scalar
product and eigenvalue error. Sections through the axis, anterior–posterior left to right.</p>
<img src="data:image/png;base64,{modes}" alt="Exact and best-matching approximate modes">

<h2>Over the whole spectrum</h2>
<img src="data:image/png;base64,{overview}" alt="Errors over the whole spectrum">
</body></html>
"""


def main():
    args = parse_args()
    output_file = args.output_file or os.path.join(args.directory, "approximation_report.html")

    load = lambda name: load_axisymmetric_harmonics(
        os.path.join(args.directory, f"{name}_harmonics.h5"), 0
    )
    exact = load(EXACT)
    approximations = {name: load(name) for name in SHOWN}
    mass = compute_mass_and_stiffness_axisymmetric(exact.mesh, compute_stiffness=False).mass

    low, middle, high = pick_modes(exact)
    with open(os.path.join(args.directory, "approximation_comparison.png"), "rb") as file:
        overview = base64.b64encode(file.read()).decode()

    html = TEMPLATE.format(
        n_modes=len(exact),
        n_points=exact.mesh.n_points,
        table=metrics_table(args.directory),
        low=", ".join(f"#{j}" for j in low),
        middle=", ".join(f"#{j}" for j in middle),
        high=", ".join(f"#{j}" for j in high),
        modes=mode_figure(exact, approximations, low + middle + high, mass),
        overview=overview,
    )
    with open(output_file, "w") as file:
        file.write(html)
    print(f"Saved {output_file}")


if __name__ == "__main__":
    main()
