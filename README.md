# RS-LSTM

Deep learning models for forecasting nuclear reactor point-kinetics behavior (neutron population/density and delayed-neutron precursor concentrations) from reactivity-insertion time series. Includes plain data-driven LSTM and FCN baselines, plus a physics-informed (PINN) LSTM variant trained with collocation points on the point-kinetics equations.

This code accompanies simulation data for **positive** reactivity insertions and **negative** reactivity insertions (split into 0-5s and 5-100s regimes).

## Repository Structure

```
RS-LSTM/
├── Dataset/
│   ├── PositiveCSVFiles/          # Positive reactivity insertion simulations
│   ├── NegativeCSVFiles_5/        # Negative reactivity, 0-5s regime
│   └── NegativeCSVFiles_5_100/    # Negative reactivity, 5-100s regime
├── data/
│   └── dataloader.py              # PyTorch Dataset/DataLoader builders (LSTM + FCN variants)
├── models/
│   └── lstm.py                    # ForecastLSTM and ForecastFCN model definitions
├── loss/
│   └── forecastLoss.py            # Data loss (MSE)
├── utils/
│   ├── visualize.py                # Training/prediction plotting utilities
│   ├── errorFile.py                # Per-variable error metrics
│   └── predictionSaving.py         # Save predictions to CSV
├── main.py                         # Train ForecastLSTM (data loss only)
├── mainFCN.py                      # Train ForecastFCN (data loss only)
├── maincollocation30.py            # Train physics-informed LSTM (PINN) with collocation points
├── eval.py                         # Evaluate a trained ForecastLSTM checkpoint
├── evalFCN.py                      # Evaluate a trained ForecastFCN checkpoint
└── evalCollocation30PINN.py        # Evaluate a trained PINN-LSTM checkpoint
```

## Data

Each simulation CSV contains a time column, a `Reactivity` (or `rho`) column, and either:

- **Positive reactivity files:** `NeutronPop, c1..c6` (neutron population and 6 precursor groups), with `Time_s` as the time column.
- **Negative reactivity files:** `NeutronDensity, C1..C6`, with `Time` as the time column.

Inputs to the models are sequences of `[time, reactivity, features...]`; targets are the features at the next timestep. Features can optionally be log10-normalized (`NORMALIZATION = 'log'`).

## Setup

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. A CUDA-capable GPU is optional (falls back to CPU automatically).

## Usage

Each training/eval script is a standalone, configuration-at-top-of-file script — edit the constants (`FILES`, `SEQ_LEN`, `EPOCHS`, `PATH`, `RUN_NAME`, etc.) before running.

**Train the LSTM baseline:**
```bash
python main.py
```

**Train the FCN baseline:**
```bash
python mainFCN.py
```

**Train the physics-informed LSTM (PINN with collocation points):**
```bash
python maincollocation30.py
```

Training scripts save a model checkpoint (`model.pth`), a training loss CSV, and a loss curve plot to `PATH/RUN_NAME/`.

**Evaluate a trained model:**
```bash
python eval.py                    # LSTM
python evalFCN.py                 # FCN
python evalCollocation30PINN.py   # PINN-LSTM
```

Eval scripts load the checkpoint from `PATH/RUN_NAME/model.pth`, run inference on held-out simulation files, write MSE/MAE/MPAE metrics to text files, save prediction-vs-target and percentage-error plots, and export predictions to CSV — all under `PATH/RUN_NAME/BalancedtestResults/`.

To switch between positive-reactivity, negative-reactivity (0-5s), and negative-reactivity (5-100s) data, comment/uncomment the corresponding `FILES` block near the top of each script.
