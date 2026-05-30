
import numpy as np


def subtract_reference(
    data: np.ndarray,
    electrode_cols: list[int],
    reference_col: int
):
    electrodes = data[:, electrode_cols]
    reference = data[:, reference_col][:, None]
    return electrodes - reference


def preprocess_eeg_array(
    data: np.ndarray,
    electrode_cols: list[int],
    reference_col: int,
):
    
    reference_eeg = subtract_reference(
        data = data,
        electrode_cols = electrode_cols,
        reference_col = reference_col
    )
    
    return reference_eeg