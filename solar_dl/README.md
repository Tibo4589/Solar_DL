# Rooftop solar panel segmentation in Flanders: code

Code for the Deep Learning project of Axel Geyskens and Tibo Van den Bergh. The step-by-step guide (the separate document) explains every step: what to run, why, and what you will be asked about in the defense. This README only tells you how the folder is organised.

## Organisation (same idea as the labs)

Every numbered script is a standalone program. It reads what the previous step left on disk, does one job, and writes its output to disk, so **state is always a file**. All data and outputs go to ONE shared folder, `SOLAR_ROOT`, which both of you can read and write. Only the code is in Git.

| Step | Script | Phase | Does |
|---|---|---|---|
| 01 | `01_check_data.py` | 1 Data | counts and pairs BDAPPV images/masks, links them to metadata |
| 02 | `02_explore_data.py` | 1 Data | class balance, panel sizes, unmasked images, area factors (m²/kWp, tilt) |
| 03 | `03_make_split.py` | 1 Data | train/val/test split **per département** (no geographic leakage) |
| 04 | `04_check_dataloader.py` | 2 U-Net | batches, augmentation picture, trivial baseline, forward pass |
| 05 | `05_train.py` | 2-4 | trains one model; every experiment is one call (`--help`) |
| 06 | `06_learning_curve.py` | 3 | IoU vs amount of data: do we need the extra (Chinese) data? |
| 07 | `07_compare_runs.py` | 3-4 | curves of several runs side by side |
| 08 | `08_choose_threshold.py` | 4 | PR curve + threshold on the validation set |
| 09 | `09_evaluate_test.py` | 4 | the ONE evaluation on the French test set |
| 10 | `10_prepare_flanders.py` | 5 Flanders | inspects orthophotos, cuts random chips to label |
| 11 | `11_rasterize_labels.py` | 5 | your QGIS polygons into masks + agreement between you two |
| 12 | `12_evaluate_flanders.py` | 5 | accuracy on Flemish chips = the domain gap |
| 13 | `13_predict_area.py` | 6 Mapping | sliding-window prediction over a whole city |
| 14 | `14_area_statistics.py` | 6 | polygons, rooftop filter, m² per city |
| 15 | `15_compare_installations.py` | 6 | detected area vs installation statistics, Leuven vs Gent |
| 16 | `16_zonnekaart_overlay.py` | 6 (optional) | suitable roofs without panels = remaining potential |

Helper modules: `common.py` (paths, **settings block**, dataset, loss, metrics, training loop), `models.py` (U-Net and SegFormer), `geo.py` (coordinates, resampling, windows).

## First time on the VSC

```bash
git clone <your repo> && cd <repo>
# edit env.sh: SOLAR_ROOT = the shared folder; edit slurm/gpu.slurm: account + partition
source env.sh                      # every session; creates the venv the first time
bash slurm/download_weights.sh     # once, on a LOGIN node: pretrained weights into SOLAR_ROOT/cache
```

Run a step on a GPU: `sbatch slurm/gpu.slurm 05_train.py --model unet --run-name unet_base`

## Try everything on fake data first

```bash
export SOLAR_ROOT=/tmp/solar_fake_$USER          # NEVER your real SOLAR_ROOT
python tools/make_fake_data.py
python 01_check_data.py && python 02_explore_data.py && python 03_make_split.py && python 04_check_dataloader.py
python 05_train.py --quick --device cpu --no-pretrained
```

All 16 steps were run end-to-end on this fake data before you got the code. That means the code runs. It does not mean the settings are right for the real data: that is your job, and it is what the guide is for.
