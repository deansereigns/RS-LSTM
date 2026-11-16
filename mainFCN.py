import os
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

import torch
from torch.optim import AdamW
import csv
from tqdm import tqdm
import numpy as np
import random
from  data.dataloader import  createFCNTimeDataLoader
from models.lstm import  ForecastFCN
from loss.forecastLoss import dataLoss
from utils.visualize import plotLosses

def seedWorker(workerId):
    workerSeed = torch.initial_seed() % 2 **32
    np.random.seed(workerSeed)
    random.seed(workerSeed)

SEED =42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
torch.use_deterministic_algorithms(True)

g = torch.Generator()
g.manual_seed(SEED)


EPOCHS = 500
BATCH_SIZE=1024
SEQ_LEN = 10
LR = 1e-5
NORMALIZATION = 'None'
PATH = "ResultsPaper/28_july_AnnalsofNuclearEnergyPaper/Positive"
RUN_NAME = "None_FCN_DataLossOnly"
TRAINING_LOSS ="trainingLoss.csv"

# Positive Reactivity Training Files
FILES=["Dataset/PositiveCSVFiles/SimulationData13.csv",
"Dataset/PositiveCSVFiles/SimulationData14.csv",
"Dataset/PositiveCSVFiles/SimulationData9.csv",
"Dataset/PositiveCSVFiles/SimulationData10.csv",
"Dataset/PositiveCSVFiles/SimulationData17.csv",
"Dataset/PositiveCSVFiles/SimulationData7.csv",
"Dataset/PositiveCSVFiles/SimulationData3.csv",
"Dataset/PositiveCSVFiles/SimulationData1.csv",
"Dataset/PositiveCSVFiles/SimulationData6.csv",
"Dataset/PositiveCSVFiles/SimulationData5.csv",
"Dataset/PositiveCSVFiles/SimulationData11.csv",
"Dataset/PositiveCSVFiles/SimulationData2.csv",
"Dataset/PositiveCSVFiles/SimulationData16.csv",
]

# Negative Reactivity 0--5s Training Files
# FILES =["Dataset/NegativeCSVFiles_5/-0.0002.csv",
# "Dataset/NegativeCSVFiles_5/-0.0006.csv",
# "Dataset/NegativeCSVFiles_5/-0.0008.csv",
# "Dataset/NegativeCSVFiles_5/-0.001.csv",
# "Dataset/NegativeCSVFiles_5/-0.002.csv",
# "Dataset/NegativeCSVFiles_5/-0.004.csv",
# "Dataset/NegativeCSVFiles_5/-0.005.csv",
# "Dataset/NegativeCSVFiles_5/-0.006.csv",
# "Dataset/NegativeCSVFiles_5/-0.007.csv",
# "Dataset/NegativeCSVFiles_5/-0.009.csv",
# "Dataset/NegativeCSVFiles_5/-0.01.csv",
# "Dataset/NegativeCSVFiles_5/-0.03.csv",
# "Dataset/NegativeCSVFiles_5/-0.04.csv",
# "Dataset/NegativeCSVFiles_5/-0.05.csv",
# "Dataset/NegativeCSVFiles_5/-0.07.csv",
# "Dataset/NegativeCSVFiles_5/-0.08.csv",
# "Dataset/NegativeCSVFiles_5/-0.09.csv"
# ]

# Negative Reactivity 5--100s Training Files
# FILES =["Dataset/NegativeCSVFiles_5_100/-0.0002.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.0006.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.0008.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.001.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.002.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.004.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.005.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.006.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.007.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.009.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.01.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.03.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.04.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.05.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.07.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.08.csv",
# "Dataset/NegativeCSVFiles_5_100/-0.09.csv"
# ]


device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device: ", device)


loader =createFCNTimeDataLoader(FILES,BATCH_SIZE,normalization=NORMALIZATION,generator =g, worker_init_fn =seedWorker)

model = ForecastFCN().to(device)
optimizer = AdamW(model.parameters(), lr=LR)

directory = os.path.join(PATH,RUN_NAME)
os.makedirs(directory, exist_ok=True)

loss_hist = [] 

lossCsvPath = os.path.join(directory,TRAINING_LOSS)

with open(lossCsvPath, mode="w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch","Loss","DataLoss","ODELoss"])

for epoch in range(EPOCHS):
    model.train()
    totalLoss=0.0
    totalDataLoss =0.0
    progressBar = tqdm(loader,desc=f"Epoch {epoch+1}/{EPOCHS}")

    for x,y in progressBar:
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
    
        preds =model(x)
        dataL = dataLoss(y,preds)
        loss= 1*dataL 
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
        optimizer.step()

        totalLoss+=loss.item()
        totalDataLoss+=dataL.item()
        progressBar.set_postfix(loss=loss.item())

    avgLoss = totalLoss/len(loader)
    avgDataLoss = totalDataLoss/len(loader)
    loss_hist.append(avgLoss)
    currentLR = optimizer.param_groups[0]["lr"]
    print(f"Epoch {epoch+1}/{EPOCHS} | AVG Loss={avgLoss:.6f} | AVG Data loss={avgDataLoss:.6f}")

    with open(lossCsvPath, mode="a", newline="") as f:
        writer =csv.writer(f)
        writer.writerow([epoch+1, avgLoss,avgDataLoss])

torch.save(model.state_dict(), os.path.join(directory,"model.pth"))
print("Model Saved Successfully")

plotLosses(loss_hist, os.path.join(directory,"trainingLoss.png"))
