import logging
import re

import h5py
import numpy as np

from embryoharmonics.celegans.embryo_model import EmbryoModel


TIME_STEP_PATTERN = re.compile(r"avg_model_(\d{3})")

_logger = logging.getLogger(__name__)


class EmbryoModelLoader:
    """
    Class for loading averaged C. elegans geometry models from an HDF5 file.
    """
    def __init__(self, file_name: str):
        """
        Initialize the embryo model loader.
        :param h5file: The HDF5 file containing the geometry models.
        """
        self.h5file = h5py.File(file_name, "r")

        # Initialize all time steps according to the given pattern
        self._time = []
        for group in self.h5file['/']:
            match = TIME_STEP_PATTERN.match(group)
            if match:
                time_step = int(match.group(1))
                self._time.append(time_step)


    @property
    def time_steps(self) -> list[int]:
        """
        Get the time steps in the HDF5 file.
        :return: The time steps in the HDF5 file
        """
        # Copy the list to avoid accidental modification
        return list(self._time)


    def load(
            self,
            time_step: int
    ) -> EmbryoModel:
        """
        Load an averaged model for the given time step.
        :param time_step: Which time step to load
        :return: An :class:`EmbryoModel` object containing the geometry data
        """
        _logger.info("Loading averaged model for time step %d from '%s'",
                     time_step, self.h5file.filename)

        try:
            model = self.h5file[f"avg_model_{time_step:03d}"]
        except KeyError as e:
            raise ValueError(f"Time step {time_step} not found in the HDF5 file") from e

        # Load metadata
        seam_cell_names = [s.decode('utf-8') for s in model["names"][:]]

        # Load central spline
        central_spline = model["central_spline"]
        central_coordinates = _convert_to_ndarray(central_spline["ordinate"][:])
        spline_domain = central_spline["abscissa"][:]

        # Load transverse splines (delimiting the worm surface)
        locations = model["transverse_splines"]
        transverse_coordinates = []
        for i in range(32):
            raw_data = locations[f"transverse_spline_{i + 1:02d}/ordinate"][:]
            cleaned_data = _convert_to_ndarray(raw_data)
            transverse_coordinates.append(cleaned_data)

        return EmbryoModel(
            seam_cell_names,
            spline_domain,
            central_coordinates,
            transverse_coordinates
        )


def _convert_to_ndarray(
        data: h5py.Dataset
) -> np.ndarray:
    return np.array([item for tup in data for item in tup[0]]).reshape(-1, 3)
