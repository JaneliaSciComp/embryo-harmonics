from typing import Dict

import numpy as np
from scipy.interpolate import CubicSpline


class EmbryoModel:
    def __init__(self, seam_cells, spline_domain, central_coordinates, transverse_coordinates):
        self.seam_cells = {name: index for index, name in enumerate(seam_cells)}
        self.spline_domain = spline_domain
        self.central_spline = CubicSpline(spline_domain, central_coordinates)
        self.transverse_splines = [CubicSpline(spline_domain, coordinates) for coordinates in transverse_coordinates]

    @property
    def n_seam_cells(self):
        return len(self.seam_cells)

    @property
    def n_transverse_splines(self):
        return len(self.transverse_splines)


def load_measurement(h5file) -> tuple[np.ndarray, np.ndarray]:
    length = h5file["measurements/length"][:]
    volume = h5file["measurements/volume"][:]
    return length, volume


def convert_to_ndarray(data):
    return np.array([item for tup in data for item in tup[0]]).reshape(-1, 3)


def load_single_model(model) -> EmbryoModel:
    names = [s.decode('utf-8') for s in model["names"][:]]

    central_spline = model["central_spline"]
    spline_domain = central_spline["abscissa"][:]
    central_coordinates = convert_to_ndarray(central_spline["ordinate"][:])

    locations = model["transverse_splines"]
    transverse_coordinates = []
    for i in range(32):
        raw_data = locations[f"transverse_spline_{i + 1:02d}/ordinate"][:]
        cleaned_data = convert_to_ndarray(raw_data)
        transverse_coordinates.append(cleaned_data)

    return EmbryoModel(names, spline_domain, central_coordinates, transverse_coordinates)


def load_avg_models(h5file, *, time_steps=None) -> Dict[int, EmbryoModel]:
    if time_steps is None:
        # there is one extra group: measurements
        time_steps = range(1, len(h5file.keys()))

    models = {}
    for time_step in time_steps:
        model = h5file[f"avg_model_{time_step:03d}"]
        models[time_step] = load_single_model(model)

    return models
