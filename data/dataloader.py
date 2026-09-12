import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ForecastDatasetWithTime(Dataset):
    """
    Dataset for sequence forecasting with time as an explicit feature.
    Input: seqLen timesteps (each with rho, time, features)
    Output: next unseen timestep (features only)
    """
    def __init__(self, fileList, seqLen=10, normalization='log'):
        self.samples = []
        self.seqLen = seqLen
        self.normalization = normalization

        for file in fileList:
            df = pd.read_csv(file)
            # --- Extract values ---
            rho = df["Reactivity"].iloc[0]
            if rho > 0:
                features= df[["NeutronPop","c1","c2","c3","c4","c5","c6"]].values# for positive
                time = df["Time_s"].values.reshape(-1, 1)    # one per row for Positive
                df.loc[df["Time_s"] < 1.0, "Reactivity"] = 0.0  # Set reactivity to 0 for t < 1s
            else:
                features = df[["NeutronDensity", "C1", "C2", "C3", "C4", "C5", "C6"]].values # for Negative
                time = df["Time"].values.reshape(-1, 1)      # one per row for Negative

            rhoVar = df["Reactivity"].values if "Reactivity" in df.columns else df["rho"].values
            # --- Normalization ---
            if normalization == 'log':
                features = np.where(features <= 0, 1e-10, features)
                features = np.log10(features)

            for i in range(len(features) - seqLen):
                xSeq = features[i:i + seqLen]
                yForecast = features[i + seqLen]
                rhoSeq = rhoVar[i:i+seqLen].reshape(-1,1)
                timeSeq = time[i:i + seqLen]             
                xFull = np.concatenate([timeSeq,rhoSeq, xSeq], axis=1)
                self.samples.append((xFull, yForecast))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        x, y = self.samples[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def createTimeDataLoader(fileList, seqLen=10, batchSize=128, shuffle=True, normalization='log',drop_last=False,generator=None,  worker_init_fn=None, pin_memory=True):
    dataset = ForecastDatasetWithTime(fileList, seqLen=seqLen, normalization=normalization)
    return DataLoader(dataset, batch_size=batchSize, shuffle=shuffle,drop_last=drop_last,generator=generator, worker_init_fn=worker_init_fn,pin_memory=pin_memory, num_workers=4)

class ForecastFCNDatasetWithTime(Dataset):
    """
    Dataset for FCN using time as a feature.
    For each timestep i we produce:
      x = [rho, time_i]
      y = features[i]

    """
    def __init__(self, fileList, normalization='log'):
        self.samples = []
        self.normalization = normalization

        for file in fileList:
            df = pd.read_csv(file)
            rho = float(df["Reactivity"].iloc[0])
            if rho > 0:
                features= df[["NeutronPop","c1","c2","c3","c4","c5","c6"]].values# for positive
                time = df["Time_s"].values.reshape(-1, 1)
                df.loc[df["Time_s"] < 1.0, "Reactivity"] = 0.0  # Set reactivity to 0 for t < 1s
            else:
                features = df[["NeutronDensity", "C1", "C2", "C3", "C4", "C5", "C6"]].values # for Negative
                time = df["Time"].values.reshape(-1, 1)

            # --- Normalization ---
            if normalization == 'log':
                features = np.where(features <= 0, 1e-10, features)
                features = np.log10(features)

            rhoVar = df["Reactivity"].values if "Reactivity" in df.columns else df["rho"].values 

            if len(features) != len(time):
                raise ValueError(f"features and time must have same number of rows (got {len(features)} vs {len(time)})")

            for i in range(len(features)):
                yForecast = features[i]              
                rho_row  = rhoVar[i].reshape(1, 1)
                time_row = time[i].reshape(1, 1)
                xFull = np.concatenate([rho_row, time_row], axis=1)
                xFull = xFull.reshape(-1)  
                self.samples.append((xFull, yForecast))

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        x, y = self.samples[index]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


def createFCNTimeDataLoader(fileList, batchSize=128, shuffle=True, normalization='log',drop_last=False,generator=None,  worker_init_fn=None, pin_memory=True):
    dataset = ForecastFCNDatasetWithTime(fileList, normalization=normalization)
    return DataLoader(dataset, batch_size=batchSize, shuffle=shuffle,drop_last=drop_last,generator=generator, worker_init_fn=worker_init_fn,pin_memory=pin_memory, num_workers=4)
