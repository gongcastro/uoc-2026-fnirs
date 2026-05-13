import itertools

import numpy as np
import pandas as pd
import xarray as xr


def find_pause_offsets(stim: pd.DataFrame, pause_code: int = 6) -> list[int]:
    """Find pause offsets.

    Args:
        stim (DataFrame): Stimuli dataframe.
        pause_code (int, optional): Trigger code assigned to pauses. Defaults to 6.

    Returns:
        list[int]: Indices of events in stimuli lists that correspond to pause offsets.
    """
    offsets = []
    for idx, (i, *_, tt) in enumerate(stim.itertuples()):
        if int(tt) == pause_code and not idx % 2:
            offsets.append(i)

    return offsets


def process_events(
    stim: pd.DataFrame,
    stim_mapping: dict[str, list[int]],
    **kwargs,
) -> pd.DataFrame:
    """Group events into their corresponding trial types.

    Arguments:
        stim (DataFrame): Stimuli dataframe.
        **kwargs: Additional arguments passed to `find_pause_offsets`.

    Returns:
        DataFrame: Stimuli dataframe with grouped trigger labels.
    """
    possible_codes = list(itertools.chain(*stim_mapping.values()))
    new_stim = stim.copy()
    lookup = {}

    pauses_remove = find_pause_offsets(new_stim, **kwargs)

    for i, *_, tt in new_stim.itertuples():
        tt = int(tt)
        if i in pauses_remove:
            new_stim.drop(index=i, inplace=True)
        if tt not in possible_codes:
            new_stim.drop(index=i, inplace=True)
        for k, v in stim_mapping.items():
            if tt in v:
                lookup[str(tt)] = str(k)

    new_stim.cd.rename_events(lookup)

    return new_stim


def _border_detrend(signal: xr.DataArray, width: int = 1):
    n = len(signal)
    if n == 0:
        return signal

    # calculate border indices
    allt = np.arange(n)
    start_indices = np.arange(0, min(width, n))
    end_indices = np.arange(max(0, n - width), n)
    normt = np.concatenate((start_indices, end_indices))

    if len(normt) < 2:
        return signal

    normdata = signal[normt]
    pcoef = np.polyfit(normt, normdata, 1)

    return signal - np.polyval(pcoef, allt)


def detrend_epochs(x: xr.DataArray, width: int = 1, dim: str = "reltime"):
    """
    Applies border detrending to epochs.

    """
    if dim not in x.dims:
        raise ValueError(f"Dimension '{dim}' not found")

    units = x.pint.units
    x = x.pint.dequantify()

    loop_dims = [d for d in x.dims if d != dim]  # all dims but time
    # stack non-time dims to a single 'combined_dim'
    # simplifies the broadcasting logic for apply_ufunc
    stacked_data = x.stack(combined_dim=loop_dims) if loop_dims else x

    out_stack = xr.apply_ufunc(
        _border_detrend,
        stacked_data.pint.dequantify(),
        input_core_dims=[[dim]],
        output_core_dims=[[dim]],
        kwargs={"width": width},
        vectorize=True,
    )

    # restore the original dimensions
    out = out_stack.unstack("combined_dim") if loop_dims else out_stack
    out = out.transpose(*x.dims)

    out = out.assign_coords({"trial_type": ("epoch", x.trial_type.to_numpy())})
    out = out.pint.quantify(units)
    return out
