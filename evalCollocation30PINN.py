import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import torch
import pandas as pd
import numpy as np
import random
from models.lstm import ForecastLSTM
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

SEQ_LEN =10
NORMALIZATION = 'log'
PATH = "Updated_5Aug_AnnalsResultsCollocation30"
RUN_NAME = "Negative0"
MODEL_SAVE_PATH = "Updated_5Aug_AnnalsResultsCollocation30/Negative0/lstm_pinn_final.pth"


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


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)

resultsDir = os.path.join(PATH,RUN_NAME,"BalancedtestResults")
os.makedirs(resultsDir, exist_ok =True)

model =ForecastLSTM()
model.load_state_dict(torch.load(MODEL_SAVE_PATH,weights_only=True))
model.eval().to(device)

error =torch.nn.MSELoss()
errorMAE = torch.nn.L1Loss()
for file in FILES:
    fileName = os.path.splitext(os.path.basename(file))[0]
    df = pd.read_csv(file)

    rho  = df["Reactivity"].iloc[0]
    if rho>0:
        features = df[["NeutronPop","c1","c2","c3","c4","c5","c6"]].values # For Positive
        df.loc[df["Time_s"] < 1.0, "Reactivity"] = 0.0  # Set reactivity to 0 for t < 1s
    else:
        features = df[["NeutronDensity","C1","C2","C3","C4","C5","C6"]].values # for Negative

    # Normalization
    if NORMALIZATION == 'log':
        features = np.where(features <= 0, 1e-10, features)
        features = np.log10(features)

    rhoVar = df["Reactivity"].values if "Reactivity" in df.columns else df["rho"].values

    inputs, targets, timeSteps, all_preds = [], [], [], []

    for i in range(20-SEQ_LEN,len(features) - SEQ_LEN):
        xSeq = features[i:i + SEQ_LEN]
        yForecast = features[i + SEQ_LEN]

        rhoSeq = rhoVar[i:i+SEQ_LEN].reshape(-1,1)

        xFull = np.concatenate([rhoSeq, xSeq], axis=1) # when time is not present as input 


        inputs.append(xFull)
        targets.append(yForecast)

        if rho>0:
            timeSteps.append(df['Time_s'].iloc[i+SEQ_LEN]) # For Positive
        else:
            timeSteps.append(df['Time'].iloc[i + SEQ_LEN])  # for Negative

    inputs =torch.tensor(inputs,dtype =torch.float32).to(device)
    targets = torch.tensor(targets,dtype=torch.float32).to(device)



    with torch.no_grad():
        y_pred = model(inputs)

    targets =targets.cpu().numpy()

    preds =y_pred[:,-1,:].cpu().numpy()

    if NORMALIZATION =='log':
        finalPred = 10**preds
        finalTargets = 10**targets
    else:
        finalPred =preds
        finalTargets =targets
    percentageErrorLog = ((preds-targets)/(targets+1e-10))*100
    percentageError = ((finalPred-finalTargets)/(finalTargets+1e-10))*100
    normMse = error(torch.tensor(preds),torch.tensor(targets)).item()
    normMae = errorMAE(torch.tensor(preds),torch.tensor(targets)).item()
    mse = error(torch.tensor(finalPred),torch.tensor(finalTargets)).item()
    mae = errorMAE(torch.tensor(finalPred),torch.tensor(finalTargets)).item()
    mpae = np.mean(np.abs(percentageError))
    print(f"Log Normalized Test MSE for {fileName}: {normMse}")
    print(f"Log Normalized Test MAE for {fileName}: {normMae}")
    print(f"Test MSE for {fileName}: {mse}")
    print(f"Test MAE for {fileName}: {mae}")
    print(f"Test MPAE for {fileName}: {mpae}")
    metricsFile= os.path.join(PATH,RUN_NAME,"BalancedtestMetrics.txt")

    perVariableMetrics = os.path.join(PATH,RUN_NAME,"BalancedperVariableMetrics.txt")
    computePerVariableMetrics(file, preds,targets,finalPred, finalTargets,perVariableMetrics)
    with open(metricsFile, mode='a') as f:
        f.write(f"Log Normalized Test MSE for {fileName}: {normMse}\n")
        f.write(f"Log Normalized Test MAE for {fileName}: {normMae}\n")
        f.write(f"Test Mean Square Error for {fileName}: {mse}\n")
        f.write(f"Test Mean Absolute Error for {fileName}: {mae}\n")
        f.write(f"Test MPAE for {fileName}: {mpae}\n")
        f.write(f"=====================\n")

    if NORMALIZATION =='log':
        plotPredictions(np.array(timeSteps),preds=preds,targets =targets,percentageError=percentageErrorLog,path=os.path.join(resultsDir,fileName), normalization='log')
    
    plotPredictions(np.array(timeSteps),preds=finalPred,targets =finalTargets,percentageError=percentageError,path=os.path.join(resultsDir,fileName), normalization='none')
        
    # Saving CSV for further reference
    savePredicitons(np.array(timeSteps),finalPred,finalTargets,percentageError,os.path.join(resultsDir,fileName),normalization='none')
    if NORMALIZATION =='log':
        savePredicitons(np.array(timeSteps),preds,targets,percentageErrorLog,os.path.join(resultsDir,fileName),normalization='log')