from typing import List
import pyvista as pv

def get_harmonic_name(i: int) -> str:
    return f"harmonic_{i:03d}"

def get_all_harmonic_names(data: pv.UnstructuredGrid) -> List[str]:
    return [name for name in data.array_names if name.startswith("harmonic_")]