# Pediatric_Spindles_HI

Code, summarized source data and tables for **Anterior Fast Spindle Duration is Associated with Both Hypopnea Burden and Attentional Performance in Children** (Haber et al., manuscript in revision).

This repository holds two things. First, what reproduces the main figures (Figures 1–6) and gives the main tables (Tables 1–3) of the paper: for each figure, the drawing script and layout and the summarized data it plots. Second, the analysis pipeline that produced those summaries from the study data. It does not contain participant-level or raw data; these are available from the corresponding author on reasonable request, as the paper's Data Availability statement describes. Supplementary figures are not included.

## Reproduce the figures

Requires [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/idossha/Pediatric_Spindles_HI.git
cd Pediatric_Spindles_HI
uv sync
./reproduce.sh       # every figure; one figure: ./reproduce.sh fig-4
```

This writes `output/fig-1.png` … `output/fig-6.png` at 300 dpi and nothing else. It works in a temporary folder: [idofig](https://github.com/idossha/idofig) runs each figure's `make.py` and composes the panels from `figure.json`, and the figure is rasterized with resvg; the folder is then deleted.

## Figures

| Paper | Script and layout | Source data | Title |
|---|---|---|---|
| Figure 1 | [`figures/fig-1`](figures/fig-1) | [`source_data/fig-1`](source_data/fig-1) | Night architecture and spindle detection |
| Figure 2 | [`figures/fig-2`](figures/fig-2) | none (schematic) | Cluster-based permutation test |
| Figure 3 | [`figures/fig-3`](figures/fig-3) | [`source_data/fig-3`](source_data/fig-3) | Normative topography of slow and fast sleep spindles |
| Figure 4 | [`figures/fig-4`](figures/fig-4) | [`source_data/fig-4`](source_data/fig-4) | Association between the hypopnea index and each spindle metric across the scalp |
| Figure 5 | [`figures/fig-5`](figures/fig-5) | [`source_data/fig-5`](source_data/fig-5) | Spindle expression and attention across the scalp |
| Figure 6 | [`figures/fig-6`](figures/fig-6) | [`source_data/fig-6`](source_data/fig-6) | Hypopnea type and fast spindle duration (exploratory) |

`make.py` draws the panels from `source_data/<id>/` only; `figure.json` places them, and `panels/<panel>/style.json` holds manual style adjustments. Each `source_data/<id>/README.md` lists its files. The scalp maps use the 172-channel montage `source_data/inside172.mat`. Children are numbered 1..N; Figures 1 and 6 show one child's night each, and Figure 6b one pair of rates per child.

## Tables

| Paper | File | Title |
|---|---|---|
| Table 1 | [`tables/Table_1.csv`](tables/Table_1.csv) | Demographic and respiratory characteristics of the sample. |
| Table 2 | [`tables/Table_2.csv`](tables/Table_2.csv) | Sleep architecture and channel-averaged sleep spindle metrics. |
| Table 3 | [`tables/Table_3.csv`](tables/Table_3.csv) | Attentional performance on the TOVA. |

## Analysis pipeline

`code/run_all.py` regenerates every table and figure input of the paper from the study data, stage by stage: verification of the spindle detections, channel-wise spindle metrics, the cluster-based permutation tests, the sensitivity analyses, the spindle–attention maps and the hypopnea-type analysis. It needs the study data (set `SPINDLEPEDS_DATA` to their location) and is included so that every analysis can be read and, with the data, rerun. See [`code/README.md`](code/README.md).

## Repository layout

```
figures/<id>/          figure.json, make.py and panel style adjustments of each main figure
source_data/<id>/      the summarized data that figure draws, with a README listing its files
tables/                Tables 1–3 as CSV
code/plotting/         the drawing modules the figure scripts share
code/detection/        spindle detection (YASA) and its checks
code/stats/            the statistics; code/run_all.py runs them in order
code/reproduce.py      what ./reproduce.sh runs
reproduce.sh           writes output/fig-N.png
```

## Citation and license

Please cite the paper (citation in [CITATION.cff](CITATION.cff); it will be updated on publication). Changes between releases are in [CHANGELOG.md](CHANGELOG.md). Code is released under the [MIT License](LICENSE).
