from typing import Iterable

import numpy as np


def encode_matlab_strings(strings: Iterable[str]) -> np.ndarray:
    """Encode a list of strings as an array of space padded ascii-chars, as
    they are stored in mat files.
    
    :param strings: The strings to encode.
    :return: The encoded strings as a numpy array.
    """
    max_length = max(len(s) for s in strings)
    encoded = [s.ljust(max_length).encode('ascii') for s in strings]
    return np.array([np.frombuffer(s, dtype=np.uint8) for s in encoded]).T.astype(np.uint16)
