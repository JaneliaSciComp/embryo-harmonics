from typing import List
import pyvista as pv

def get_harmonic_name(i: int) -> str:
    return f"harmonic_{i:03d}"

def get_all_harmonic_names(data: pv.UnstructuredGrid) -> List[str]:
    return [name for name in data.array_names if name.startswith("harmonic_")]

def retain_harmonics(data: pv.DataSet) -> pv.DataSet:
    for name in data.array_names:
        if not name.startswith("harmonic_"):
            data.point_data.remove(name)
    return data