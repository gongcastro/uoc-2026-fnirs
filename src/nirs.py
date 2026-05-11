import itertools

import pandas as pd


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
