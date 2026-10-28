# Building an Echoshader application

Echoshader provides a generic, read-only [Panel](https://panel.holoviz.org/) application for exploring calibrated acoustic datasets. It combines reusable visualization elements with a shared ping-time selection, channel and color controls, and vessel-track views where geographic positions are available.

The current application has a fixed HoloViz layout configured with a small YAML file. YAML selects the dataset and sets a few navigation/display options; it does not define an arbitrary graph of visualization elements, controls, or interactions. Future developments will allow application-specific features to be added through optional extensions, enabled and configured through YAML.

## Requirements

The default app expects an xarray Dataset containing `Sv`, a `channel` coordinate, and an increasing `datetime64` `ping_time` coordinate. The `Sv` variable has dimensions `(channel, ping_time, vertical)`, where `vertical` is `range_sample`, `depth`, or `echo_range`. Physical vertical coordinates such as `depth` and `echo_range` should be provided where appropriate. `frequency_nominal`, `latitude`, and `longitude` enable additional labeling and navigation features when available.

The app accepts existing **Zarr** or **NetCDF** datasets. Acoustic processing, calibration, navigation interpolation, and geometric resampling belong upstream, for example in Echopype. The app does not silently modify the acoustic data to make datasets plot-compatible.

## Start with the example notebook

Run the [element example notebook](https://github.com/echostack-org/echoshader/blob/main/examples/notebooks/element_examples.ipynb) from `examples/notebooks/`. It first processes one EK80 RAW file to demonstrate echograms and geometric resampling, then downloads the processed survey dataset used by the remaining elements and the example app.

The download cell extracts `MVBS.zip` into `MVBS.zarr` in the notebook working directory. Alternatively, download [MVBS.zip](https://github.com/echostack-org/echoshader/releases/download/example-data-v1/MVBS.zip) directly and extract it into `examples/notebooks/`. No RAW processing is required to launch the app with this prepared dataset.

The example contains NOAA SH2306 Pacific hake survey data from August 12, 2023, 12:45–23:59 UTC: five frequencies (18, 38, 70, 120, and 200 kHz), 675 one-minute time bins, and 501 depth coordinates spanning 0–500 m. It was processed with Echopype 0.12.0, with 1-metre depth bins, a 9.8 m depth offset, and navigation interpolation. See the [dataset release](https://github.com/echostack-org/echoshader/releases/tag/example-data-v1) for provenance.

## YAML configuration

The repository includes `examples/app/app_config.yaml`:

```yaml
app:
  title: Echoshader - Application template
  source: survey  # Dataset key under data
  route: basic  # URL path: /basic (use "" for /)

data:
  survey:
    type: zarr
    path: ../notebooks/MVBS.zarr  # Relative to this YAML file
    label: NOAA SH2306 Pacific hake survey  # Sidebar name

navigation:
  max_pings: 500  # Maximum time samples per selection

display:
  max_range_samples: 1200  # Maximum vertical samples displayed
  map_height: 500  # Map height in pixels
```

For your own data, change `path` to an existing local Zarr store. To use NetCDF, set `type: netcdf` and provide the corresponding file path. Relative data paths are resolved relative to the YAML file when launched through the application launcher.

`max_pings` limits the displayed ping-time window. For MVBS, these samples are time bins rather than original instrument pings. `map_height` sets the map height in pixels; its width follows the available layout column. `max_range_samples` limits vertical samples in supported dimensions; it is a display sampling control, not a transformation of the stored data.

## Launching the application

From the repository root:

```bash
python -m echoshader.app.launcher examples/app/app_config.yaml --port 1805
```

Open <http://localhost:1805/basic>. The `app.route` setting supplies `/basic`; use an empty route to serve the app at `/`.

From Python, also from the repository root:

```python
from echoshader.app import launch

launch("examples/app/app_config.yaml", port=1805, address="127.0.0.1", show=True)
```

Alternatively, build the Panel template without serving it:

```python
from echoshader.app.builder import build_app
from echoshader.app.config import load_config

config = load_config("examples/app/app_config.yaml")
app = build_app(config, config_dir="examples/app")
```

The returned object is a `panel.template.FastListTemplate`. The launcher starts a Panel server. The explicit `port` argument (or CLI `--port`) takes precedence over `app.port` in YAML; otherwise the launcher defaults to port 5006.

## Interaction and display

The application offers a selected ping-time interval, linked date controls and vessel-track navigation when positions exist, single-channel and multi-frequency echograms, and a three-channel RGB view. Changing a selection requests the corresponding subset of the dataset. The slider updates plots when released. Map box or tap selection requests the interval between the earliest and latest selected time samples. If that interval exceeds `max_pings`, the app uses a centered window capped at the configured limit. The blue track highlights the active time interval, which can include positions outside the original geographic box. Date controls, the slider, the track highlight, and echograms share this interval.

Multi-frequency plots offer a colormap selector, including Echopype's EK500 palette, and Sv colour limits. Tricolor uses the selected red, green, and blue acoustic channels rather than a colormap.

A tricolor echogram requires three channels on a compatible physical vertical grid. The native Sv data from different transducers may have different physical sample geometries even if they share `range_sample` indices. Use a physically aligned dataset for tricolor visualization; do not treat sample numbers as meters.

## Validation and troubleshooting

Echoshader validates acoustic dataset structure and checks tricolor compatibility separately. An ordinary echogram may be valid even when three channels cannot be combined into an RGB plot.

- **No vessel track:** check for finite `latitude` and `longitude` values aligned with `ping_time`.
- **Mostly empty MVBS:** check whether your dataset includes time bins without recordings; the app preserves missing periods.
- **Tricolor geometry mismatch:** select a common physical geometry dataset, rather than forcing native channels onto one sample-index axis.
- **Slow rendering:** reduce the displayed ping window or prepare an appropriate coarser upstream product. Chunk sizing and display windows affect performance.

See [Visualization elements](elements.md) for standalone use.
