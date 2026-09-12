import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
import json
import os


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

COLLOCATION_PKL = "Updated_5Aug_AnnalsResultsCollocation30/Negative0/collocation_points.pkl"
MODEL_SAVE_PATH = "Updated_5Aug_AnnalsResultsCollocation30/Negative0/lstm_pinn_final.pth"
RESULTS_DIR = "Updated_5Aug_AnnalsResultsCollocation30/Negative0"
os.makedirs(RESULTS_DIR, exist_ok=True)

EPOCHS = 5000
EPOCHS_DATA = 100       
BATCH_SIZE = 1024
SEQ_LEN = 10
N_BASE_POINTS = 100     
N_SYNTHETIC = 30        
LR = 1e-5

# PKE physical constants
BETA_I  = torch.tensor([0.000285, 0.00139, 0.00125, 0.00273, 0.00088, 0.00021])
LAMBDA_I= torch.tensor([0.0127,   0.0317,  0.118,   0.311,   1.40,    3.87])
BETA    = 0.00675
ELL     = 1e-4
LN10    = 2.30258509299

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
# "Dataset/NegativeCSVFiles_5/-0.001.csv",
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


def generate_base_collocations(file_list, seq_len=SEQ_LEN, n_base=N_BASE_POINTS):
    """
    For each file, pick N_BASE_POINTS evenly spaced indices as base collocation points.
    Each base point at index i uses real data from i-seq_len to i-1 as input,
    and real data at index i as the ground truth target.
    """
    collocation_dict = {}
    for file_path in file_list:
        df = pd.read_csv(file_path)
        n_points = len(df)
        # Valid range: need seq_len steps before, so start from seq_len
        valid_indices = np.arange(seq_len, n_points)
        if len(valid_indices) == 0:
            continue
        base_points = np.linspace(valid_indices[0], valid_indices[-1], n_base, dtype=int)
        collocation_dict[os.path.basename(file_path)] = np.unique(base_points).tolist()

    with open(COLLOCATION_PKL, "wb") as f:
        pickle.dump(collocation_dict, f)
    print(f"Collocation points saved to {COLLOCATION_PKL}")


class BaseCollocationDataset(Dataset):
    """
    Dataset for base collocation points.
    Each sample:
      x      : [seq_len, 8]  — rho concatenated with log10(features) for indices i-seq_len to i-1
      y      : [7]           — log10(features) at index i  (ground truth)
      dt     : scalar        — time step from index i-1 to i
    """
    def __init__(self, file_list, seq_len=SEQ_LEN, collocation_pkl_path=COLLOCATION_PKL, is_test=False):
        self.samples = []

        if not is_test:
            with open(collocation_pkl_path, "rb") as f:
                collocation_points = pickle.load(f)
        else:
            collocation_points = {}

        for file_path in file_list:
            df = pd.read_csv(file_path)
            rho = df["Reactivity"].iloc[0] if "Reactivity" in df.columns else df["rho"].iloc[0]
            if rho > 0:
                features = df[["NeutronPop", "c1", "c2", "c3", "c4", "c5", "c6"]].values
                time_raw = df["Time_s"].values
                df.loc[df["Time_s"] < 1.0, "Reactivity"] = 0.0  # Set reactivity to 0 for t < 1s
            else:
                features = df[["NeutronDensity", "C1", "C2", "C3", "C4", "C5", "C6"]].values
                time_raw = df["Time"].values

            
            rhoVar = df["Reactivity"].values if "Reactivity" in df.columns else df["rho"].values

            features = np.where(features <= 0, 1e-10, features)
            features = np.log10(features)

            fname = os.path.basename(file_path)
            indices = np.arange(seq_len, len(df)) if is_test else collocation_points.get(fname, [])

            for i in indices:
                x_seq = features[i - seq_len:i]                          
                y_gt  = features[i]                                      
                rho_seq = rhoVar[i-seq_len:i].reshape(-1,1)
                x_full  = np.concatenate([rho_seq, x_seq], axis=1) 
                dt = float(time_raw[i] - time_raw[i - 1])

                self.samples.append((x_full, y_gt, dt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, dt = self.samples[idx]
        return (torch.tensor(x,  dtype=torch.float32),
                torch.tensor(y,  dtype=torch.float32),
                torch.tensor(dt, dtype=torch.float32))


class SyntheticCollocationDataset(Dataset):
    """
    Dataset for synthetic (inserted) collocation points.

    For each base collocation point i (with real input window [i-seq_len, i-1]):
      - Distribute N_SYNTHETIC time points across the 9 gaps inside the window.
        e.g. for seq_len=10, gaps are between indices:
             (i-10,i-9), (i-9,i-8), ..., (i-2,i-1)   → 9 gaps
        30 points spread as evenly as possible across 9 gaps.

    Each synthetic point k has:
      - t_k   : interpolated timestamp (between two real consecutive steps)
      - rho   : same constant reactivity from the file
      - input : the 10 real steps just before t_k  → the shifted real window
    """
    def __init__(self, file_list, seq_len=SEQ_LEN, n_synthetic=N_SYNTHETIC,
                 collocation_pkl_path=COLLOCATION_PKL):
        self.samples = []

        with open(collocation_pkl_path, "rb") as f:
            collocation_points = pickle.load(f)

        for file_path in file_list:
            df = pd.read_csv(file_path)
            rho = df["Reactivity"].iloc[0] if "Reactivity" in df.columns else df["rho"].iloc[0]

            if rho > 0:
                features = df[["NeutronPop", "c1", "c2", "c3", "c4", "c5", "c6"]].values
                time_raw = df["Time_s"].values
                df.loc[df["Time_s"] < 1.0, "Reactivity"] = 0.0 
            else:
                features = df[["NeutronDensity", "C1", "C2", "C3", "C4", "C5", "C6"]].values
                time_raw = df["Time"].values

            rhoVar = df["Reactivity"].values if "Reactivity" in df.columns else df["rho"].values
            features = np.where(features <= 0, 1e-10, features)
            features = np.log10(features)

            fname = os.path.basename(file_path)
            base_indices = collocation_points.get(fname, [])

            
            n_gaps = seq_len - 1  
            counts = np.zeros(n_gaps, dtype=int)
            counts[:] = n_synthetic // n_gaps
            remainder = n_synthetic % n_gaps
            counts[:remainder] += 1 

            for i in base_indices:
                window_start = i - seq_len 
                syn_inputs = []   
                syn_dt     = []   
                t_prev = time_raw[window_start]

                for gap_idx in range(n_gaps):
                    t_left  = time_raw[window_start + gap_idx]
                    t_right = time_raw[window_start + gap_idx + 1]
                    n_pts   = counts[gap_idx]

                    if n_pts == 0:
                        continue

                    syn_times = np.linspace(t_left, t_right, n_pts + 2)[1:-1]

                    for t_k in syn_times:
                        dt_k = float(t_k - t_prev)
                        t_prev = t_k

                        real_end_idx = window_start + gap_idx + 1  
                        real_start_idx = real_end_idx - seq_len

                        if real_start_idx < 0:
                            pad_len = -real_start_idx
                            x_seq = np.concatenate([
                                np.tile(features[0], (pad_len, 1)),
                                features[0:real_end_idx]
                            ], axis=0)
                            rho_seq = np.concatenate([
                                np.full(pad_len, rhoVar[0]),
                                rhoVar[0:real_end_idx]
                            ], axis=0).reshape(-1, 1)
                        else:
                            x_seq   = features[real_start_idx:real_end_idx]           
                            rho_seq = rhoVar[real_start_idx:real_end_idx].reshape(-1, 1)

                        rho_seq = rho_seq.astype(np.float32)
                        x_full  = np.concatenate([rho_seq, x_seq], axis=1)

                        syn_inputs.append(x_full)
                        syn_dt.append(dt_k)

                if len(syn_inputs) == 0:
                    continue

                # Pad or trim to exactly N_SYNTHETIC
                while len(syn_inputs) < n_synthetic:
                    syn_inputs.append(syn_inputs[-1])
                    syn_dt.append(syn_dt[-1])
                syn_inputs = syn_inputs[:n_synthetic]
                syn_dt     = syn_dt[:n_synthetic]

                self.samples.append((
                    np.stack(syn_inputs, axis=0).astype(np.float32),  
                    np.array(syn_dt, dtype=np.float32),        
                    float(rho)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        syn_inputs, syn_dt, rho = self.samples[idx]
        return (torch.tensor(syn_inputs, dtype=torch.float32),  
                torch.tensor(syn_dt,     dtype=torch.float32),  
                torch.tensor(rho,        dtype=torch.float32))  


class ForecastLSTM(nn.Module):
    def __init__(self, input_dim=8, hidden_dim=128, output_dim=7, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x: [batch, seq_len, 8]
        returns: [batch, 7]
        """
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def data_loss(preds, targets):
    return (preds - targets).pow(2).mean()


def ode_loss_single_step(pred, prev_state, dt, rho, device):
    """
    Finite difference ODE residual for ONE step.
    pred       : [batch, 7]  —> log10 prediction at current synthetic point
    prev_state : [batch, 7]  —> log10 state at previous point (real or synthetic prediction)
    dt         : [batch]     —> time step
    rho        : [batch]     —> reactivity
    """
    beta_i   = BETA_I.to(device)
    lambda_i = LAMBDA_I.to(device)

    dt = dt.view(-1, 1).clamp(min=1e-8)
    rho = rho.view(-1, 1)

    u_k, v_k     = pred[:, 0:1],       pred[:, 1:7]
    u_prev, v_prev = prev_state[:, 0:1], prev_state[:, 1:7]

    du_dt_fd  = (u_k - u_prev) / dt
    dv_dt_fd  = (v_k - v_prev) / dt

    du_dt_ode = (1.0 / LN10) * (
        ((rho - BETA) / ELL) +
        torch.sum(lambda_i * torch.pow(10, v_prev - u_prev), dim=1, keepdim=True)
    )
    dv_dt_ode = (1.0 / LN10) * (
        (beta_i / ELL).unsqueeze(0) * torch.pow(10, u_prev - v_prev) - lambda_i
    )

    res_u = (du_dt_fd - du_dt_ode).pow(2).mean()
    res_v = (dv_dt_fd - dv_dt_ode).pow(2).mean()
    return res_u + res_v

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = ForecastLSTM().to(device)
    optimizer = AdamW(model.parameters(), lr=LR)

    # Build datasets
    base_dataset = BaseCollocationDataset(FILES)
    base_loader  = DataLoader(base_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    syn_dataset  = SyntheticCollocationDataset(FILES)
    syn_loader   = DataLoader(syn_dataset, batch_size=max(1, BATCH_SIZE // N_SYNTHETIC),
                              shuffle=True, num_workers=4, pin_memory=True)

    print(f"Base collocation samples : {len(base_dataset)}")
    print(f"Synthetic sample groups  : {len(syn_dataset)}  ({len(syn_dataset)*N_SYNTHETIC} total synthetic points)")
    print(f"Training on {device}...")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    loss_history = []   

    epoch_pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

    for epoch in epoch_pbar:
        model.train()
        total_data_l   = 0.0
        total_ode_l    = 0.0
        n_base_batches = 0
        n_syn_batches  = 0

        if epoch < EPOCHS_DATA:
            phase = "Phase1- DataOnly"
            base_pbar = tqdm(base_loader, desc=f"  [Base]  Ep {epoch+1}", leave=False, unit="batch")
            for x, y, dt in base_pbar:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()

                preds  = model(x)
                d_loss = data_loss(preds, y)

                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_data_l   += d_loss.item()
                n_base_batches += 1
                base_pbar.set_postfix(data_loss=f"{d_loss.item():.5f}")

        else:
            phase = "Phase2- ODEOnly"
            syn_pbar = tqdm(syn_loader, desc=f"  [Syn]   Ep {epoch+1}", leave=False, unit="batch")
            for syn_inputs, syn_dt, rho in syn_pbar:
                syn_inputs = syn_inputs.to(device)
                syn_dt     = syn_dt.to(device)
                rho_batch  = rho.to(device)

                optimizer.zero_grad()
                total_ode_step = torch.tensor(0.0, device=device)

                prev_state = syn_inputs[:, 0, -1, 1:]

                for k in range(N_SYNTHETIC):
                    x_k    = syn_inputs[:, k, :, :]
                    dt_k   = syn_dt[:, k]
                    pred_k = model(x_k)

                    step_ode       = ode_loss_single_step(pred_k, prev_state, dt_k, rho_batch, device)
                    total_ode_step = total_ode_step + step_ode
                    prev_state     = pred_k.detach()

                ode_l = total_ode_step / N_SYNTHETIC
                ode_l.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_ode_l   += ode_l.item()
                n_syn_batches += 1
                syn_pbar.set_postfix(ode_loss=f"{ode_l.item():.5f}")

        avg_data = total_data_l / max(n_base_batches, 1)
        avg_ode  = total_ode_l  / max(n_syn_batches,  1)

        epoch_pbar.set_postfix(phase=phase, data=f"{avg_data:.5f}", ode=f"{avg_ode:.5f}")

        loss_history.append({
            "epoch":     epoch + 1,
            "phase":     phase,
            "data_loss": avg_data,
            "ode_loss":  avg_ode,
        })

        history_json = os.path.join(RESULTS_DIR, "training_loss_history.json")
        with open(history_json, "w") as f:
            json.dump(loss_history, f, indent=2)
        history_csv = os.path.join(RESULTS_DIR, "training_loss_history.csv")
        pd.DataFrame(loss_history).to_csv(history_csv, index=False)

        tqdm.write(
            f"Epoch {epoch+1}/{EPOCHS} [{phase}] | "
            f"Data Loss: {avg_data:.6f} | ODE Loss: {avg_ode:.6f}"
        )

    
    history_json = os.path.join(RESULTS_DIR, "training_loss_history.json")
    with open(history_json, "w") as f:
        json.dump(loss_history, f, indent=2)
    pd.DataFrame(loss_history).to_csv(
        os.path.join(RESULTS_DIR, "training_loss_history.csv"), index=False
    )

    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")
    print(f"Loss history saved to {history_json}")


if __name__ == "__main__":
    if not os.path.exists(COLLOCATION_PKL):
        generate_base_collocations(FILES)
    train_model()