
import numpy as np
import mne


def subtract_reference(
    data: np.ndarray,
    electrode_cols: list[int],
    reference_col: int
):
    electrodes = data[:, electrode_cols]
    reference = data[:, reference_col][:, None]
    return electrodes - reference


def to_mne_raw_array(
    eeg_data: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    units: str
):
    eeg_data = np.asarray(eeg_data, dtype=np.float64)
    
    if eeg_data.ndim != 2:
        raise ValueError("EEG data must be 2D. (n_timesteps, n_channels)")

    n_channels = eeg_data.shape[1]
    
    if len(ch_names) != n_channels:
        raise ValueError("Length of ch_names must match number of EEG channels")

    if units == "uV":
        eeg_data = eeg_data * 1e-6
    elif units == "mV":
        eeg_data = eeg_data * 1e-3
    elif units == "V":
        pass
    else:
        raise ValueError("units must be 'uV', 'mV', or 'V'")
    
    eeg_mne = eeg_data.T
    
    info = mne.create_info(
        ch_names = ch_names,
        sfreq = sfreq,
        ch_types = ['eeg'] * n_channels
    )
    
    raw = mne.io.RawArray(eeg_mne, info, verbose = False)
    return raw

def preprocess_raw_mne(
    raw: mne.io.RawArray,
    l_freq: float,
    h_freq: float,
    notch_freq: float | None
):
    
    raw = raw.copy()
    
    raw.filter(
        l_freq = l_freq,
        h_freq = h_freq,
        verbose = False
    )
    
    if notch_freq is not None:
        raw.notch_filter(
            freqs = [notch_freq],
            verbose = False
        )
    
    return raw

def raw_to_numpy(
    raw: mne.io.RawArray,
    output_units: str
):
    data = raw.get_data().T
    
    if output_units == "uV":
        data = data * 1e6
    elif output_units == "mV":
        data = data * 1e3
    elif output_units == "V":
        pass
    else:
        raise ValueError("units must be 'uV', 'mV', or 'V'")
    
    return data


def preprocess_eeg_array_with_mne(
    data: np.ndarray,
    sfreq: float,
    electrode_cols: list[int],
    reference_col: int,
    ch_names: list[str],
    input_units: str,
    output_units: str,
    l_freq: float,
    h_freq: float,
    notch_freq: float | None
):
    
    reference_eeg = subtract_reference(
        data = data,
        electrode_cols = electrode_cols,
        reference_col = reference_col
    )
    
    raw = to_mne_raw_array(
        eeg_data = reference_eeg,
        sfreq = sfreq,
        ch_names = ch_names,
        units = input_units
    )
    
    raw_clean = preprocess_raw_mne(
        raw = raw,
        l_freq = l_freq,
        h_freq = h_freq,
        notch_freq = notch_freq
    )
    
    clean_array = raw_to_numpy(
        raw = raw_clean,
        output_units = output_units
    )
    
    return clean_array