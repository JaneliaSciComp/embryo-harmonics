import re
from typing import List

import pyvista as pv

HARMONIC_FORMAT = "harmonic_{:03d}"
HARMONIC_PATTERN = re.compile(r"harmonic_(\d{3})")

def harmonic_name(i: int) -> str:
    return f"harmonic_{i:03d}"


def all_harmonic_names(data: pv.DataSet) -> List[str]:
    return [name for name in data.array_names if HARMONIC_PATTERN.fullmatch(name)]


def retain_harmonics(data: pv.DataSet) -> pv.DataSet:
    for name in data.array_names:
        if not HARMONIC_PATTERN.fullmatch(name):
            data.point_data.remove(name)
    return data
