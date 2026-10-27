# Stochastic seismic inversion with pseudo-wells: ODiSI and CNN

Code to reproduce the results of

> N. T. Tung, N. D. Lam, T. Q. Minh, P. T. Cuong, P. T. H. Ha, H. Q. Dat, D. H. Hien,
> *Stochastic seismic inversion using a large number of pseudo-wells and convolutional neural
> networks*, submitted to Computers & Geosciences.

Two inversions of one 2D seismic line, using extended elastic impedance (EEI) at
χ = −60, −7.5, 25 and 45°. Both use the same pseudo-well prior: Markov-chain facies
(Table 1 of the manuscript) and the rock-physics relations of Table 2 in two-way time.
AR(4) fluctuations and the Vshale of each facies are calibrated on Well 2X.

* **ODiSI.** For every trace, 15,000 pseudo-wells are simulated and matched to the field
  EEI, and the best 50 are kept. Each property comes from the angle it is most sensitive to
  at Well 2X: Vp −7.5°, Vs −60°, ρ 25°, porosity and Vshale 45°.
* **CNN.** Two networks are trained on 100,000 pseudo-wells drawn from the same prior.
  * A regression network maps EEI to Vp, Vs, ρ, porosity and Vshale. Its uncertainty
    comes from Monte Carlo dropout.
  * A classification network maps EEI to facies (shale, silt, sand, calcareous sand),
    with probabilities.

## Data

The field data are proprietary. This package contains only what the inversions need
(`Data/README.md`):

* the AVO intercept and gradient of the line, with anonymised headers;
* the live window of the line;
* the wavelet;
* the Well 2X calibration of the prior, as statistics only.

The seismic stack and the well logs of Wells 2X and 9P are **not** distributed. Without
them, the following parts of the code still exist but cannot run:

* the calibration of the prior (`ODiSI_libs/calibrate_well2x.py`). Its output,
  `Data/rock_physics_2x.json`, is included;
* the Well 2X figures (`ODiSI_libs/make_well2x_figures.py`);
* the blind-well comparison at Well 9P: the well panels, correlation coefficients,
  RMSE and bias, and facies scores.

Every inversion result runs without the logs: the training set, ODiSI, both networks,
the field sections, the ablation and the noise test. The scripts skip the well
comparison when the logs are absent.

## Layout

```
Data/                    AVO intercept/gradient, live window, wavelet, Well 2X calibration
ODiSI_libs/
  pseudowell/            Markov-chain facies, rock physics (Table 2, AR(4), Gassmann), EEI,
                         Butterworth filter, SEG-Y I/O, plotting
  odisi_inversion.py     per-trace ODiSI (prior and matching settings in CFG)
  make_paper_figures.py  sections, calculated EEI, Well 9P panel (logs only if available)
  calibrate_well2x.py    calibration of the prior on Well 2X (needs the Well 2X log)
  make_well2x_figures.py Figures 2 and 3 (needs the Well 2X log)
CNN_libs/
  make_datamat.py        training set: pseudo-wells from the ODiSI prior and their EEI
  cnn_model.py           network architecture (Figure 9)
  cnn_common.py          windows, field EEI from A/B, stretching onto the training window
  train_cnn.py           regression network
  infer_cnn.py           regression network on the field line, MC dropout -> SEG-Y
  classify_facies.py     facies-classification network (training and field line)
  make_training_figures.py  training curve, pseudo-well examples, correlations, confusion matrices
  eei_angles.py          EEI of the same pseudo-wells at other angles (ablation)
  noise_sensitivity.py   test pseudo-wells with noise on the AVO gradient
Notebooks/
  01_Running_ODiSI.ipynb           ODiSI at the four angles and the multi-angle result (Figures 7, 8)
  02_CNN_4chis_seed0.ipynb         training set, regression and facies networks, field line (Figures 10-15)
  03_CNN_chi_m60.ipynb             regression network with EEI at -60 deg only     (Table 5)
  04_CNN_chi_m7.5.ipynb            regression network with EEI at -7.5 deg only    (Table 5)
  05_CNN_chi_25.ipynb              regression network with EEI at 25 deg only      (Table 5)
  06_CNN_chi_45.ipynb              regression network with EEI at 45 deg only      (Table 5)
  07_CNN_25chis_m60_to_60.ipynb    25 angles, -60 to 60 deg in steps of 5 deg; summary of Table 5 and noise test
  nb_utils.py                      paths, running the library scripts, plotting helpers
```

## Installation

You need Python 3.10 or later. A GPU is recommended for the CNN; it also runs on a CPU, but
slowly.

```
pip install -r requirements.txt     # numpy scipy matplotlib numba joblib tqdm segyio torch jupyterlab
```

## Running

The notebooks in `Notebooks/` run the whole study step by step and show the figures of the
manuscript inline. They are saved **with their outputs**, so the results can be inspected
without running anything. To run them again:

```
cd Notebooks
jupyter lab                     # or: jupyter notebook
```

Run notebook 01 (ODiSI, ~4 min on 16 CPU cores) and notebook 02 (CNN, ~50 min on one H100 GPU)
first. Notebooks 03-07 (angle tests, ~30 min each on one H100) reuse the training set of
notebook 02 and can be run in any order; the last part of notebook 07 summarises the tests that
have been run. Results that already exist are reused; set `FORCE = True` in notebook 01 to rerun
ODiSI.

Each step calls a script of `ODiSI_libs/` or `CNN_libs/` with the same arguments as on the
command line, writes the full log next to its outputs, and prints the last lines. The scripts
can also be used directly; quick test, a few minutes on a CPU:

```
cd ODiSI_libs && python odisi_inversion.py --chi 45 --nsim 500 --traces 0:16 --no-plots --outdir /tmp/odisi_test
cd ../CNN_libs && python make_datamat.py --n 2000 --outdir /tmp/td
python train_cnn.py --train-dir /tmp/td --outdir /tmp/reg --arch keras --epochs 2 --reg-loss mse \
    --cls-weight 0 --calibration none --gain-jitter 0 --mc-test 0
python infer_cnn.py --run /tmp/reg --field-mode stretch --mc 5 --tag test
```

## Outputs

| folder | content |
|---|---|
| `Notebooks/outputs/ODiSI/chi_<χ>/` | ODiSI at one angle: `<prop>_{mean,std,q25,q75}_*.segy`, facies and facies probabilities, QC figures |
| `Notebooks/outputs/ODiSI/multichi_well2x/` | **ODiSI result of the manuscript**: each property from its angle; facies from 45° |
| `Notebooks/outputs/CNN/training_data/` | 100,000 pseudo-wells: EEI, Vp, Vs, ρ, porosity, Vshale, facies |
| `Notebooks/outputs/CNN/regression_s0/` | regression network; `field/`: SEG-Y of the mean, std and quantiles; `figures/`: manuscript figures |
| `Notebooks/outputs/CNN/facies_s0/` | facies network: facies and probability SEG-Y, confusion matrix, test scores |
| `Notebooks/outputs/CNN/ablation_<name>/` | angle tests: `chi_m60`, `chi_m7.5`, `chi_25`, `chi_45`, `eei25` |
| `Notebooks/outputs/CNN/noise_sensitivity.json` | noise test of notebook 07 |

`Notebooks/outputs/` is created when the notebooks run. It holds about 3 GB (mostly the training
sets) and is not part of the distributed package.

All SEG-Y outputs are on the 1 ms grid of the line and can be compared trace by trace.

## Settings

| | ODiSI | CNN |
|---|---|---|
| prior | `CFG` in `odisi_inversion.py`: Table 1 transition matrices and bed thicknesses; Table 2 relations with AR(4) fluctuations (`pseudowell/rock_physics.py`, `Data/rock_physics_2x.json`); brine | same prior (`make_datamat.py`) |
| EEI | Butterworth 3–60 Hz | same |
| field EEI | 0.65 · (A cos χ + B sin χ) convolved with the wavelet | same, with the ODiSI window resampled onto the training window and standardised per angle |
| matching / training | 15,000 pseudo-wells per trace; amplitude scale calibrated on the line; static shifts ±4 ms; misfit per macro layer; best 50 | regression: MSE, 150 epochs, batch 32, AdamW with one-cycle schedule, maximum learning rate 1e-3, dropout 0.3. Facies: cross-entropy, 60 epochs, batch 256, time-stretch, gain and noise augmentation, class decision weights tuned on validation pseudo-wells. Split 80/10/10 % by pseudo-well |
| uncertainty | mean, std and quartiles of the best 50; facies probabilities | MC dropout: 100 passes (regression), 50 passes (facies) |

ODiSI is deterministic and reproduces the manuscript exactly. CNN training and MC-dropout
inference on a GPU are not bit-reproducible, so CNN values change slightly from run to run.

## Results in the manuscript

The Well 9P comparison (Table 3) needs the confidential well log. For reference, the
correlation coefficients at Well 9P over 2.886–3.016 s are:

| | Vp | Vs | ρ | Vshale | φ | mean |
|---|---|---|---|---|---|---|
| ODiSI, multi-χ | 0.681 | 0.763 | 0.684 | 0.651 | 0.630 | 0.682 |
| CNN | 0.699 | 0.780 | 0.704 | 0.731 | 0.674 | 0.718 |

These results can be checked without well data:

* the median correlation of the regression network on the 10,000 test pseudo-wells:
  0.84 / 0.84 / 0.82 / 0.80 / 0.77 for Vp / Vs / ρ / porosity / Vshale
  (notebook 02);
* the facies confusion matrices on the training and test pseudo-wells;
* the ODiSI and CNN sections.

## License

The code is released under the MIT License (`LICENSE`). The data in `Data/` are provided only
to reproduce the results of the manuscript.
