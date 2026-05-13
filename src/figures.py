from pathlib import Path

from cedalion import io
from cedalion import units
from cedalion.vis.colors import threshold_cmap, mask_cmap, segmented_cmap
from cedalion.sigproc import frequency
from cedalion.sigproc import quality
from cedalion.sigproc import motion_correct
from cedalion.nirs import cw
import cedalion.vis.blocks as vbx

from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy import signal
import xarray as xr

from src import nirs

PATH: Path = Path("img")
DATA_PATH: Path = Path("data")
DPI: int = 800
CHROMO_COLORS = {760: "tab:blue", 850: "tab:red", "HbO": "tab:red", "HbR": "tab:blue"}
SCI_THRESH: float = 0.60
PSP_THRESH: float = 0.04
PROP_CLEAN: float = 0.70
SSP_STIM_MAPPING: dict[str, list[int]] = {
    "A": [9],
    "B": [10],
    "Exclude": [2, 4, 6, 7],
}


def mask_cmap2(true_is_good=True):
    """A red and blue colormap to color binary masks."""

    colors = ["#DC3220", "#DC3220", "#0C7BDC", "#0C7BDC"]
    if not true_is_good:
        colors = colors[::-1]

    norm, cmap = segmented_cmap(
        "mask_cmap",
        vmin=0,
        vmax=1.0,
        segments=zip([0, 0.5, 0.5, 1], colors),
        bad="white",
        over="white",
        under="white",
    )

    return norm, cmap


rec_good = io.read_snirf(DATA_PATH / "adult.snirf")[0]
rec_bad = io.read_snirf(DATA_PATH / "mop-breastfeeding.snirf")[0]
rec_ugly = io.read_snirf(DATA_PATH / "ssp-good.snirf")[0]

rec_ugly.stim = nirs.process_events(rec_ugly.stim, SSP_STIM_MAPPING)
rec_ugly.stim = rec_ugly.stim[rec_ugly.stim["trial_type"] != "Exclude"]

##### Hb absorption coefficient ##########################

df = pd.read_csv(Path("data", "absorption.tsv"), sep="\t")
df = df[(df.wavelength >= 650) & (df.wavelength <= 900)]

fig, ax = plt.subplots(1, 1)
fig.set_size_inches(7, 4)
ax.plot(df.wavelength, df.hbo, c="tab:red", label="HbO", lw=2)
ax.plot(df.wavelength, df.hbr, c="tab:blue", label="HbR", lw=2)
ax.axvspan(690, 760, lw=0, color="tab:blue", alpha=1 / 4)
ax.axvspan(830, 900, lw=0, color="tab:red", alpha=1 / 4)
ax.set_ylabel("Molar Extinction Coefficient (cm$^{-1}$/M)")
ax.set_xlabel("Wavelenth [$\\lambda$] (nm)")
fig.legend(ncol=2)
fig.savefig(PATH / "hb-absorption.png", dpi=DPI)

#### good amp 1 ########################################

amp = rec_good["amp"]

fig, axes = plt.subplots(2, 1)

for wl_i, wl in enumerate(amp.wavelength.values):
    for ch in amp.channel.values:
        d = amp.sel(wavelength=wl, channel=ch)
        axes[wl_i].plot(d.time, d, c="k", alpha=0.5)
fig.legend(ncols=2)
fig.savefig(PATH / "good-amp-1.png", dpi=DPI)

#### good amp 2 ########################################

fig, axes = plt.subplots(2, 1, sharey=True)

for wl_i, wl in enumerate(amp.wavelength.values):
    ax = axes[wl_i]

    for ch in amp.channel.values:
        d = amp.sel(wavelength=wl, channel=ch)
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, alpha=0.5)
        ax.set_ylabel(f"{wl:.0f} nm")

        if wl_i == 0:
            ax.set_xticks([])
_ = fig.supylabel("Amplitude (V)")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)

fig.savefig(PATH / "good-amp-2.png", dpi=DPI)

#### good amp 4 ########################################

channels = amp.channel.values[:1]
fig, ax = plt.subplots(len(channels), 1)

ax.set_title(ch)

for wl_i, wl in enumerate(amp.wavelength.values):
    d = amp.sel(wavelength=wl, channel=ch)
    ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=f"{wl:.0f} nm")

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "good-amp-3.png", dpi=DPI)

channels = amp.channel.values[:3]
times = (100, 130)

fig, axes = plt.subplots(len(channels), 1)
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(amp.wavelength.values):
        d = amp.sel(wavelength=wl, channel=ch, time=slice(*times))
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "good-amp-4.png", dpi=DPI)


#### good od 1 ########################################

rec_good["od"] = cw.int2od(rec_good["amp"])
od = rec_good["od"]

fig, axes = plt.subplots(2, 1, sharey=True)
for wl_i, wl in enumerate(od.wavelength.values):
    ax = axes[wl_i]

    for ch in od.channel.values:
        d = od.sel(wavelength=wl, channel=ch)
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, alpha=0.5)
        ax.set_ylabel(f"{wl:.0f} nm")

        if wl_i == 0:
            ax.set_xticks([])
_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)

fig.savefig(PATH / "good-od-1.png", dpi=DPI)

#### good od 2 #####################################

channels = od.channel.values[:1]
fig, ax = plt.subplots(len(channels), 1)

ax.set_title(ch)

for wl_i, wl in enumerate(od.wavelength.values):
    d = od.sel(wavelength=wl, channel=ch)
    ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=f"{wl:.0f} nm")

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "good-od-2.png", dpi=DPI)

#### good od 3 #####################################

channels = od.channel.values[:3]
times = (100, 130)

fig, axes = plt.subplots(len(channels), 1)
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch, time=slice(*times))
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "good-od-3.png", dpi=DPI)

#### bad 1 #####################################

rec_bad["od"] = cw.int2od(rec_bad["amp"])
od = rec_bad["od"]

fig, axes = plt.subplots(2, 1, sharey=True)
for wl_i, wl in enumerate(od.wavelength.values):
    ax = axes[wl_i]

    for ch in od.channel.values:
        d = od.sel(wavelength=wl, channel=ch)
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, alpha=0.5)
        ax.set_ylabel(f"{wl:.0f} nm")

        if wl_i == 0:
            ax.set_xticks([])
_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.savefig(PATH / "bad-od-1.png", dpi=DPI)

#### bad 2 #####################################

channels = od.channel.values[:3]
fig, axes = plt.subplots(len(channels), 1)

for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "bad-od-2.png", dpi=DPI)

#### bad 3 #######################################

channels = od.channel.values[:1]
fig, ax = plt.subplots(len(channels), 1)

ax.set_title(ch)

for wl_i, wl in enumerate(od.wavelength.values):
    d = od.sel(wavelength=wl, channel=ch)
    ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=f"{wl:.0f} nm")

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "bad-od-3.png", dpi=DPI)

#### bad od 4 #####################################

channels = od.channel.values[:3]
times = (100, 130)

fig, axes = plt.subplots(len(channels), 1)
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch, time=slice(*times))
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "bad-od-4.png", dpi=DPI)

#### ugly 2 #####################################

rec_ugly["od"] = cw.int2od(rec_ugly["amp"])
od = rec_ugly["od"]

fig, axes = plt.subplots(2, 1, sharey=True)
for wl_i, wl in enumerate(od.wavelength.values):
    ax = axes[wl_i]

    for ch in od.channel.values:
        d = od.sel(wavelength=wl, channel=ch)
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, alpha=0.5)
        ax.set_ylabel(f"{wl:.0f} nm")

        if wl_i == 0:
            ax.set_xticks([])
_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.savefig(PATH / "ugly-od-1.png", dpi=DPI)

#### ugly 2 #####################################

channels = od.channel.values[:3]
fig, axes = plt.subplots(len(channels), 1)

for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "ugly-od-2.png", dpi=DPI)

#### bad 3 #######################################

channels = od.channel.values[:1]
fig, ax = plt.subplots(len(channels), 1)

ax.set_title(ch)

for wl_i, wl in enumerate(od.wavelength.values):
    d = od.sel(wavelength=wl, channel=ch)
    ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=f"{wl:.0f} nm")

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "ugly-od-3.png", dpi=DPI)

#### ugly od 4 #####################################

channels = od.channel.values[:3]
times = (100, 130)

fig, axes = plt.subplots(len(channels), 1)
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch, time=slice(*times))
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "ugly-od-4.png", dpi=DPI)

#### scalp coupling index 1 ############################

fig, (ax_signal, ax_sci) = plt.subplots(2, 1, height_ratios=(0.4, 0.6))
od = rec_ugly["od"]
for ch_i, ch in enumerate(od.channel.values):
    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = None
        if ch_i == 0:
            label = f"{wl:.0f} nm"
        ax_signal.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label, alpha=0.5)

ax_signal.set_ylabel("Optical density")
ax_signal.set_yticks([])
ax_signal.set_yticklabels([])

window_length = 5 * units.s
sci, sci_mask = quality.sci(od, window_length, SCI_THRESH)

sci_norm, sci_cmap = threshold_cmap("sci_cmap", 0, 1, SCI_THRESH)
sci_binary_norm, sci_binary_cmap = mask_cmap()

m = ax_sci.pcolormesh(
    sci.time,
    np.arange(len(sci.channel)),
    sci,
    shading="nearest",
    norm=sci_norm,
    cmap=sci_cmap,
)
cb = plt.colorbar(m, ax=ax_sci, location="bottom", shrink=0.3)
ax_signal.set_xlim(0, od.time.max())
ax_sci.set_xlim(0, od.time.max())
ax_sci.set_ylabel("Channel")
ax_sci.set_xlabel("Time (s)")

ax_sci.tick_params(axis="y", labelsize=7)
ax_sci.yaxis.set_ticks(np.arange(len(sci.channel)))
ax_sci.yaxis.set_ticklabels(sci.channel.values)

fig.tight_layout()
fig.legend(ncols=2)
fig.savefig(PATH / "sci-1.png", dpi=DPI)

#### peak spectral power 1 ############################

fig, (ax_signal, ax_psp) = plt.subplots(2, 1, height_ratios=(0.4, 0.6))
for ch_i, ch in enumerate(od.channel.values):
    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = None
        if ch_i == 0:
            label = f"{wl:.0f} nm"
        ax_signal.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label, alpha=0.5)

ax_signal.set_ylabel("Optical density")
ax_signal.set_yticks([])
ax_signal.set_yticklabels([])

psp, psp_mask = quality.psp(od, window_length, PSP_THRESH)

psp_norm, psp_cmap = threshold_cmap("psp_cmap", 0, 0.7, PSP_THRESH)
psp_binary_norm, psp_binary_cmap = mask_cmap()

m = ax_psp.pcolormesh(
    psp.time,
    np.arange(len(psp.channel)),
    psp,
    shading="nearest",
    norm=psp_norm,
    cmap=psp_cmap,
)
cb = plt.colorbar(m, ax=ax_psp, location="bottom", shrink=0.3)
ax_signal.set_xlim(0, od.time.max())
ax_psp.set_xlim(0, od.time.max())
ax_psp.set_ylabel("Channel")
ax_psp.set_xlabel("Time (s)")

ax_psp.tick_params(axis="y", labelsize=7)
ax_psp.yaxis.set_ticks(np.arange(len(psp.channel)))
ax_psp.yaxis.set_ticklabels(psp.channel.values)

fig.tight_layout()
fig.legend(ncols=2)
fig.savefig(PATH / "psp-1.png", dpi=DPI)

#### sci-psp 3 ######################################

fig, (ax_signal, ax_mask) = plt.subplots(2, 1, height_ratios=(0.4, 0.6))
for ch_i, ch in enumerate(od.channel.values):
    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = None
        if ch_i == 0:
            label = f"{wl:.0f} nm"
        ax_signal.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label, alpha=0.5)

ax_signal.set_ylabel("Optical density")
ax_signal.set_yticks([])
ax_signal.set_yticklabels([])

all_mask = sci_mask & psp_mask
all_binary_norm, all_binary_cmap = mask_cmap()

m = ax_mask.pcolormesh(
    all_mask.time,
    np.arange(len(all_mask.channel)),
    all_mask,
    shading="nearest",
    norm=all_binary_norm,
    cmap=all_binary_cmap,
)
cb = plt.colorbar(m, ax=ax_mask, location="bottom", shrink=0.3)
ax_signal.set_xlim(0, od.time.max())
ax_mask.set_xlim(0, od.time.max())
ax_mask.set_ylabel("Channel")
ax_mask.set_xlabel("Time (s)")

ax_mask.tick_params(axis="y", labelsize=7)
ax_mask.yaxis.set_ticks(np.arange(len(all_mask.channel)))
ax_mask.yaxis.set_ticklabels(all_mask.channel.values)

fig.tight_layout()
fig.legend(ncols=2)
fig.savefig(PATH / "all-mask-1.png", dpi=DPI)

#### sci-psp 4 ######################################

prop_time_clean = all_mask.sum(dim="time") / len(all_mask.time)
pruned_channels = prop_time_clean[prop_time_clean < PROP_CLEAN].channel.values

fig, (ax_signal, ax_prop) = plt.subplots(2, 1, height_ratios=(0.4, 0.6))
for ch_i, ch in enumerate(od.channel.values):
    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = None
        if ch_i == 0:
            label = f"{wl:.0f} nm"
        alpha = 0.1 if ch not in pruned_channels else 1
        ax_signal.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label, alpha=alpha)

ax_signal.set_ylabel("Optical density")
ax_signal.set_yticks([])
ax_signal.set_yticklabels([])

all_mask = sci_mask & psp_mask
all_binary_norm, all_binary_cmap = mask_cmap()

m = ax_prop.pcolormesh(
    all_mask.time,
    np.arange(len(all_mask.channel)),
    all_mask,
    shading="nearest",
    norm=all_binary_norm,
    cmap=all_binary_cmap,
)

# Create a mask: True for rows to mask (whiten out), False for rows to highlight
mask = np.ones_like(all_mask, dtype=bool)
r = [i for i, ch in enumerate(all_mask.channel.values) if ch in pruned_channels]
mask[r, :] = 0
masked_data = np.ma.masked_where(mask, all_mask)
_, all_binary_cmap = mask_cmap2()

m2 = ax_prop.pcolormesh(
    all_mask.time,
    np.arange(len(all_mask.channel)),
    masked_data,
    # color="white",
    shading="nearest",
    norm=all_binary_norm,
    cmap=all_binary_cmap,
)

m2.set_alpha(0.9)
cb = plt.colorbar(m, ax=ax_prop, location="bottom", shrink=0.3)

ax_signal.set_xlim(0, od.time.max())
ax_prop.set_xlim(0, od.time.max())
ax_prop.set_ylabel("Channel")
ax_prop.set_xlabel("Time (s)")

ax_prop.tick_params(axis="y", labelsize=7)
ax_prop.yaxis.set_ticks(np.arange(len(all_mask.channel)))
ax_prop.yaxis.set_ticklabels(all_mask.channel.values)

fig.tight_layout()
fig.legend(ncols=2)
fig.savefig(PATH / "prop-mask-1.png", dpi=DPI)

#### pruned ################################################

quality_mask = [prop_time_clean >= 0.7]
od, drop_list = quality.prune_ch(rec_ugly["od"], quality_mask, "all")

fig, ax = plt.subplots(1, 1)
for ch_i, ch in enumerate(od.channel.values):
    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = None
        if ch_i == 0:
            label = f"{wl:.0f} nm"
        ax.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label, alpha=0.5)


_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "pruned-1.png", dpi=DPI)

#### motion arctifact attenuation 1 #######################

od_ma = motion_correct.tddr(motion_correct.wavelet(od))
rec_ugly["od"] = od_ma
channels = od_ma.channel.values[:1]

fig, (ax_raw, ax_ma) = plt.subplots(2, len(channels))
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax_raw.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)
        d = od_ma.sel(wavelength=wl, channel=ch)
        ax_ma.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1)
    ax_raw.set_title(f"{ch} before TDDR + Wavelet")
    ax_ma.set_title(f"{ch} after TDDR + Wavelet")

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "tddr-1.png", dpi=DPI)

#### motion arctifact attenuation 2 #######################

channels = od_ma.channel.values[-3:-2]

fig, (ax_raw, ax_ma) = plt.subplots(2, len(channels))
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for wl_i, wl in enumerate(od.wavelength.values):
        d = od.sel(wavelength=wl, channel=ch)
        label = f"{wl:.0f} nm" if ch_i == 0 else None
        ax_raw.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1, label=label)
        d = od_ma.sel(wavelength=wl, channel=ch)
        ax_ma.plot(d.time, d, c=CHROMO_COLORS[wl], lw=1)
    ax_raw.set_title(f"{ch} before TDDR + Wavelet")
    ax_ma.set_title(f"{ch} after TDDR + Wavelet")

_ = fig.supylabel("Optical density")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "tddr-2.png", dpi=DPI)

#### concentrations #######################

dpf_vals = [4.2, 5.3]
dpf_coords = {"wavelength": rec_ugly["od"].wavelength}
dpf = xr.DataArray(dpf_vals, dims="wavelength", coords=dpf_coords)
rec_ugly["conc"] = cw.od2conc(rec_ugly["od"], rec_ugly.geo3d, dpf)

conc = rec_ugly["conc"]

fig, ax = plt.subplots(1, 1)
for ch_i, ch in enumerate(conc.channel.values):
    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch)
        label = None
        if ch_i == 0:
            label = chromo
        ax.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label, alpha=0.5)

_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "conc-1.png", dpi=DPI)


#### concentrations ###########################

channels = conc.channel.values[:3]

fig, axes = plt.subplots(len(channels), 1)
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch)
        label = chromo if chromo_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label)
    ax.set_title(ch)

_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "conc-2.png", dpi=DPI)


#### concentrations ###########################

channels = conc.channel.values[:3]

fig, axes = plt.subplots(len(channels), 1)
for ch_i, ch in enumerate(channels):
    ax = axes[ch_i]
    ax.set_title(ch)
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch, time=slice(*times))
        label = chromo if ch_i == 0 else None
        ax.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label)
    ax.set_title(ch)

_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "conc-3.png", dpi=DPI)

#### csd 1 ##########################################

fmin, fmax = 0.02, 0.5
conc_filt = conc.cd.freq_filter(
    fmin=fmin * units.Hz, fmax=fmax * units.Hz, butter_order=5
)
rec_ugly["conc"] = conc_filt

time = conc.time.values
n_samples = len(time)
chromo = "HbO"
fs = frequency.sampling_rate(conc)
csd_data = conc.sel(chromo=chromo)
csd_data_filt = conc_filt.sel(chromo=chromo)

f, p = signal.csd(time, csd_data.to_numpy(), fs=fs.magnitude, scaling="density")
f_filt, p_filt = signal.csd(
    time, csd_data_filt.to_numpy(), fs=fs.magnitude, scaling="density"
)

fig, (ax_raw, ax_filt) = plt.subplots(2, 1)
for ch_i, ch in enumerate(csd_data.channel.values):
    ax_raw.loglog(f, np.abs(p[ch_i]), color=CHROMO_COLORS[chromo], alpha=0.5)
    ax_filt.loglog(f_filt, np.abs(p_filt[ch_i]), color=CHROMO_COLORS[chromo], alpha=0.5)

for ax in (ax_raw, ax_filt):
    ax.hlines([0.01, 0.01], [0.5, 1.17], [1, 3.17], linewidths=[2], colors=["k", "k"])
    ax.text(x=(0.5 + 1) / 2, y=0.05, s="Respiration", ha="center", va="bottom")
    ax.text(x=(1.17 + 3.17) / 2, y=0.05, s="Heartbeat", ha="center", va="bottom")

    ax.axvspan(0.02, 0.5, alpha=0.25, lw=0, color="lightgrey")
    ax.set_xticks([0, 1e-10, 0.1, 1.17, 3.17, fs.magnitude])
    ax.set_xlim(0, fs.magnitude)
    ax.xaxis.set_major_formatter(ScalarFormatter())

_ = fig.supxlabel("Frequency (Hz)")
_ = fig.supylabel("CSD Magnitude in $$\Delta$μM/L^2/Hz$")
ax_raw.set_title("Before band-pass filtering")
ax_filt.set_title("After band-pass filtering")

fig.savefig(PATH / "csd-1.png", dpi=DPI)

#### csd 2 ##########################################

channels = conc.channel.values[:1]

fig, (ax_before, ax_after) = plt.subplots(2, len(channels))
for ch_i, ch in enumerate(channels):
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch)
        label = chromo if ch_i == 0 else None
        ax_before.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label)
        d = conc_filt.sel(chromo=chromo, channel=ch)
        ax_after.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1)

ax_before.set_title("Before frequency filtering")
ax_after.set_title("After frequency filtering")
fig.suptitle(channels[0])
_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "csd-2.png", dpi=DPI)


#### csd 3 ##########################################

channels = conc.channel.values[:1]

fig, (ax_before, ax_after) = plt.subplots(2, len(channels), sharey=True)
for ch_i, ch in enumerate(channels):
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch, time=slice(*times))
        label = chromo if ch_i == 0 else None
        ax_before.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label)
        d = conc_filt.sel(chromo=chromo, channel=ch, time=slice(*times))
        ax_after.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1)

ax_before.set_title("Before frequency filtering")
ax_after.set_title("After frequency filtering")
fig.suptitle(channels[0])
_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "csd-3.png", dpi=DPI)

#### csd 4 ##########################################

channels = conc.channel.values[-3:-2]

fig, (ax_before, ax_after) = plt.subplots(2, len(channels))
for ch_i, ch in enumerate(channels):
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch)
        label = chromo if ch_i == 0 else None
        ax_before.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label)
        d = conc_filt.sel(chromo=chromo, channel=ch)
        ax_after.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1)

ax_before.set_title("Before frequency filtering")
ax_after.set_title("After frequency filtering")
fig.suptitle(channels[0])
_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "csd-4.png", dpi=DPI)


#### csd 5 ##########################################

channels = conc.channel.values[-3:-2]

fig, (ax_before, ax_after) = plt.subplots(2, len(channels), sharey=True)
for ch_i, ch in enumerate(channels):
    if ch_i != (len(channels) - 1):
        ax.set_xticks([])

    for chromo_i, chromo in enumerate(conc.chromo.values):
        d = conc.sel(chromo=chromo, channel=ch, time=slice(*times))
        label = chromo if ch_i == 0 else None
        ax_before.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1, label=label)
        d = conc_filt.sel(chromo=chromo, channel=ch, time=slice(*times))
        ax_after.plot(d.time, d, c=CHROMO_COLORS[chromo], lw=1)

ax_before.set_title("Before frequency filtering")
ax_after.set_title("After frequency filtering")
fig.suptitle(channels[0])
_ = fig.supylabel("$\Delta$μM/L")
_ = fig.supxlabel("Time (s)")
fig.legend(ncols=2)
fig.savefig(PATH / "csd-5.png", dpi=DPI)


#### epochs 1 ###############################################

fig, ax = plt.subplots(1, 1)
channels = conc_filt.channel.values[:1]
for i, ch in enumerate(channels):
    for chromo in conc_filt.chromo.values:
        ax.plot(
            conc_filt.time,
            conc_filt.sel(channel=ch, chromo=chromo),
            CHROMO_COLORS[chromo],
            label=chromo,
        )
    ax.set_title(f"{ch}")
    # add stim markers using Cedalion's plot_stim_markers function
    vbx.plot_stim_markers(ax, rec_ugly.stim, y=1)
    ax.set_ylabel(r"$\Delta$μM/L")

ax.legend(ncol=6)
ax.set_label("Time (s)")
# ax.set_xlim(0, 100)
fig.tight_layout()
fig.savefig(PATH / "epochs-1.png", dpi=DPI)


#### epochs 2 ###############################################

times = 100, 260
fig, ax = plt.subplots(1, 1)
channels = conc_filt.channel.values[:1]
for i, ch in enumerate(channels):
    for chromo in conc_filt.chromo.values:
        d = conc_filt.sel(channel=ch, chromo=chromo, time=slice(*times))
        ax.plot(d.time, d, CHROMO_COLORS[chromo], label=chromo)
    ax.set_title(f"{ch}")
    # add stim markers using Cedalion's plot_stim_markers function
    vbx.plot_stim_markers(
        ax,
        rec_ugly.stim[
            (rec_ugly.stim["onset"] <= times[1]) & (times[0] <= rec_ugly.stim["onset"])
        ],
        y=1,
    )
    ax.set_ylabel(r"$\Delta$μM/L")

ax.legend(ncol=6)
ax.set_label("Time (s)")
# ax.set_xlim(0, 100)
fig.tight_layout()
fig.savefig(PATH / "epochs-2.png", dpi=DPI)

#### epochs 3 ###############################################

epochs = rec_ugly["conc"].cd.to_epochs(
    rec_ugly.stim,  # stimulus dataframe
    ["A", "B"],  # select fingertapping events, discard others
    before=5 * units.s,  # seconds before stimulus
    after=30 * units.s,  # seconds after stimulus
)
# calculate baseline
baseline = epochs.sel(reltime=(epochs.reltime < 0)).mean("reltime")

# subtract baseline
epochs = epochs - baseline
epochs = nirs.detrend_epochs(epochs)
hrf = epochs.groupby("trial_type").mean("epoch")


fig, axes = plt.subplots(1, 2, sharey=True)
for t_i, trial_type in enumerate(hrf.trial_type.values):
    for chromo_i, chromo in enumerate(hrf.chromo.values):
        for channel in hrf.channel.values:
            axes[t_i].hlines([0], [-5], [30], linestyles=["dotted"], colors=["k"])
            d = hrf.sel(channel=channel, trial_type=trial_type, chromo=chromo)
            d_se = hrf.sel(channel=channel, trial_type=trial_type, chromo=chromo)
            axes[t_i].plot(d.reltime, d, c=CHROMO_COLORS[chromo], alpha=0.15)
        d = hrf.sel(trial_type=trial_type, chromo=chromo).mean("channel")

        label = chromo if t_i == 0 else None
        axes[t_i].plot(d.reltime, d, lw=2, color=CHROMO_COLORS[chromo], label=label)
    axes[t_i].set_title(f"Trial type {trial_type}")
    axes[t_i].axvspan(0, 13, color="lightgrey", linewidth=0, alpha=0.5)


fig.legend(ncol=2)
fig.supxlabel("Time (s)")
fig.supylabel(r"$\Delta$μM/L")
fig.tight_layout()
fig.savefig(PATH / "epochs-3.png", dpi=DPI)
