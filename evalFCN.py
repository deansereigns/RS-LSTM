import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import os
import pandas as pd
import numpy as np
import random
from models.lstm import ForecastFCN 
from utils.visualize import plotPredictions
from utils.predictionSaving import savePredicitons
from utils.errorFile import computePerVariableMetrics

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic =True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)


NORMALIZATION = 'None'
PATH = "ResultsPaper/28_july_AnnalsofNuclearEnergyPaper/Positive"
RUN_NAME = "None_FCN_DataLossOnly"

# Positive Reactivity Test Files
FILES=["Dataset/PositiveCSVFiles/SimulationData4.csv",
       "Dataset/PositiveCSVFiles/SimulationData8.csv",
       "Dataset/PositiveCSVFiles/SimulationData12.csv",
       "Dataset/PositiveCSVFiles/SimulationData15.csv"
   
]

# Negative Reactivity 0--5s Test Files
# FILES = [
#     "Dataset/NegativeCSVFiles_5/-0.0004.csv",
#     "Dataset/NegativeCSVFiles_5/-0.003.csv",
#     "Dataset/NegativeCSVFiles_5/-0.008.csv",
#     "Dataset/NegativeCSVFiles_5/-0.02.csv",
#     "Dataset/NegativeCSVFiles_5/-0.06.csv",
#     "Dataset/NegativeCSVFiles_5/-0.1.csv",
# ]

# Negative Reactivity 5--100s Test Files
# FILES = [
#     "Dataset/NegativeCSVFiles_5_100/-0.0004.csv",
#     "Dataset/NegativeCSVFiles_5_100/-0.003.csv",
#     "Dataset/NegativeCSVFiles_5_100/-0.008.csv",
#     "Dataset/NegativeCSVFiles_5_100/-0.02.csv",
#     "Dataset/NegativeCSVFiles_5_100/-0.06.csv",
#     "Dataset/NegativeCSVFiles_5_100/-0.1.csv",
# ]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)

modelPath = os.path.join(PATH, RUN_NAME, "model.pth")
resultsDir = os.path.join(PATH, RUN_NAME, "BalancedtestResults")
os.makedirs(resultsDir, exist_ok=True)

model = ForecastFCN()
model.load_state_dict(torch.load(modelPath, weights_only=True))
model.eval().to(device)
error = torch.nn.MSELoss()
errorMAE = torch.nn.L1Loss()

for file in FILES:
    fileName = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)

    # Reactivity (file-level scalar)
    rho = df["Reactivity"].iloc[0]

    
    if rho > 0:
        features = df[["NeutronPop","c1","c2","c3","c4","c5","c6"]].values # For Positive
        df.loc[df["Time_s"] < 1.0, "Reactivity"] = 0.0  # Set reactivity to 0 for t < 1s
    else:
        features = df[["NeutronDensity","C1","C2","C3","C4","C5","C6"]].values # for Negative

    # Handle both possible time column names
    if "Time_s" in df.columns:
        time = df["Time_s"].values
    else:
        time = df["Time"].values

    rhoVar = df["Reactivity"].values if "Reactivity" in df.columns else df["rho"].values
    # Normalization (log)
    if NORMALIZATION == 'log':
        features = np.where(features <= 0, 1e-10, features)
        features = np.log10(features)

    # Build per-timestep FCN inputs
    # For FCN we create one input row per timestep (e.g. [rho, time_i])
    inputs = []
    targets = []
    timeSteps = []

    
    for i in range(20,len(features)):
        # target at timestep i is features[i] (7-dim)
        yForecast = features[i]

        # rho is scalar per file -> make row
        rho_row = np.array([rhoVar[i]])
        time_val = np.array([time[i]])     # shape (1,)

        # Final input row: [rho, time]
        xFull = np.concatenate([rho_row, time_val], axis=0)  # shape (2,)

        inputs.append(xFull)
        targets.append(yForecast)
        timeSteps.append(time[i])

    # Convert to tensors and to device
    inputs = torch.tensor(np.stack(inputs), dtype=torch.float32).to(device)   # shape (n_samples, inputDim)
    targets = torch.tensor(np.stack(targets), dtype=torch.float32).to(device) # shape (n_samples, outputDim)

    # Run inference
    with torch.no_grad():
        y_pred = model(inputs)   # expected shape (n_samples, outputDim)

    preds = y_pred.cpu().numpy()
    targets_np = targets.cpu().numpy()

    # If normalization was 'log', preds/targets are in log10-space — convert back for linear metrics/plots
    if NORMALIZATION == 'log':
        finalPred = np.power(10.0, preds)
        finalTargets = np.power(10.0, targets_np)
    else:
        finalPred = preds
        finalTargets = targets_np

    # percentage errors: compute both in model-space (log) and in linear-space
    # model-space percentage error (only meaningful if NORMALIZATION == 'log')
    if NORMALIZATION == 'log':
        percentageErrorLog = ((preds - targets_np) / (np.abs(targets_np) + 1e-10)) * 100.0
    else:
        percentageErrorLog = ((finalPred - finalTargets) / (np.abs(finalTargets) + 1e-10)) * 100.0

    # Original space percentage error
    percentageError = ((finalPred - finalTargets) / (np.abs(finalTargets) + 1e-10)) * 100.0

    normMse = error(torch.tensor(preds),torch.tensor(targets_np)).item()
    normMae = errorMAE(torch.tensor(preds),torch.tensor(targets_np)).item()
    mse = error(torch.tensor(finalPred),torch.tensor(finalTargets)).item()
    mae = errorMAE(torch.tensor(finalPred),torch.tensor(finalTargets)).item()
    mpae = np.mean(np.abs(percentageError))
    print(f"Log Normalized Test MSE for {fileName}: {normMse}")
    print(f"Log Normalized Test MAE for {fileName}: {normMae}")
    print(f"Test MSE for {fileName}: {mse}")
    print(f"Test MAE for {fileName}: {mae}")
    print(f"Test MPAE for {fileName}: {mpae}%")
    metricsFile= os.path.join(PATH,RUN_NAME,"BalancedtestMetrics.txt")


    perVariableMetrics = os.path.join(PATH,RUN_NAME,"BalancedperVariableMetrics.txt")
    computePerVariableMetrics(file, preds,targets_np,finalPred, finalTargets,perVariableMetrics)
    with open(metricsFile, mode='a') as f:
        f.write(f"Log Normalized Test MSE for {fileName}: {normMse}\n")
        f.write(f"Log Normalized Test MAE for {fileName}: {normMae}\n")
        f.write(f"Test Mean Square Error for {fileName}: {mse}\n")
        f.write(f"Test Mean Absolute Error for {fileName}: {mae}\n")
        f.write(f"Test MPAE for {fileName}: {mpae}%\n")
        f.write(f"=====================\n")

    if NORMALIZATION == 'log':
        plotPredictions(np.array(timeSteps), preds=preds, targets=targets_np, percentageError=percentageErrorLog,
                        path=os.path.join(resultsDir, fileName), normalization='log')

    plotPredictions(np.array(timeSteps), preds=finalPred, targets=finalTargets, percentageError=percentageError,
                    path=os.path.join(resultsDir, fileName), normalization='none')

    # Save CSVs (linear)
    savePredicitons(np.array(timeSteps), finalPred, finalTargets, percentageError,
                    os.path.join(resultsDir, fileName), normalization='none')

    # Save model-space preds if using log normalization
    if NORMALIZATION == 'log':
        savePredicitons(np.array(timeSteps), preds, targets_np, percentageErrorLog,
                        os.path.join(resultsDir, fileName), normalization='log')

print("FCN inference complete.")
