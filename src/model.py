import os
import copy
import torch
import numpy    as np
import torch.nn as nn
import torch.nn.functional as F

from typing           import Optional, Dict, Tuple
from sklearn.metrics  import r2_score
from sklearn.metrics  import mean_absolute_error
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats      import pearsonr, spearmanr, kendalltau, wasserstein_distance

from src.configs      import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ProtCNN(nn.Module):
    def __init__(self, feat_dim: int, hidden_channels: int, depth_conv: int, depth_ffnn: int, dropout_rate: float)->None:
        super().__init__()
        self.gelu = nn.GELU()
        self.conv_layers = nn.ModuleList(
            [
                self._make_conv_block(
                    feat_dim if i == 0 else hidden_channels,
                    hidden_channels,
                    first=(i == 0),
                )
                for i in range(depth_conv)
            ]
        )

        ff_layers = nn.ModuleList()
        in_dim = hidden_channels * 2

        for _ in range(depth_ffnn - 1):
            out_dim = in_dim // 2
            ff_layers.append(
                self._make_ff_block(
                    in_dim,
                    out_dim,
                    dropout_rate,
                )
            )
            in_dim = out_dim

        self.ff_layers = nn.Sequential(*ff_layers)
        self.regressor = nn.Linear(in_dim, 1)
     
        self._init_weights()

    def _make_conv_block(self, in_ch: int, out_ch: int, first: bool=False)->nn.Sequential:
        conv = (
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=7,
                padding=3,
                dilation=1,
                bias=False,
            )
            if first
            else nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=5,
                padding=6,
                dilation=3,
                bias=False,
            )
        )

        return nn.Sequential(
            conv,
            nn.GroupNorm(8, out_ch),
            nn.GELU(),
            nn.Dropout1d(p=0.15),
        )
    
    def _make_ff_block(self, in_dim: int, out_dim: int, dropout_rate: float)->nn.Sequential:
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout_rate),
        )
    
    def _init_weights(self)->None:
        """
        Initialize trainable layer weights.

        Applies Kaiming normal initialization to all convolutional and
        fully connected layer weights and initializes biases to zero.
        """
        
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor)->torch.Tensor:
        """
        Forward pass through the model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feat_dim, sequence_length).

        Returns
        -------
        torch.Tensor
            Regression output tensor of shape (batch_size, 1).
        """
        
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            
            if x.shape == residual.shape:
                x = x + residual

        x = self.gelu(x)

        x_avg = F.adaptive_avg_pool1d(x, 1)
        x_max = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)

        x = self.ff_layers(x)
        x_reg = self.regressor(x)
      
        return x_reg
   
    def extract_embedding_last(self, x: torch.Tensor)->torch.Tensor:
        """
        Extracts last embedding produced from the last layer before the
        regression head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feat_dim, sequence_length).

        Returns
        -------
        torch.Tensor
             Tensor containing the embedding produced from the last layer before the
             regression head.
        """
        for conv in self.conv_layers:
            residual = x
            x = conv(x)
            
            if x.shape == residual.shape:
                x = x + residual

        x = self.gelu(x)

        x_avg = F.adaptive_avg_pool1d(x, 1)
        x_max = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)

        x = self.ff_layers(x)
        return x
    
    def extract_embedding_all(self, x: torch.Tensor)->torch.Tensor:
        """
         Extract intermediate embeddings from the network before the
        regression head.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, feat_dim, sequence_length).

        Returns
        -------
        torch.Tensor
            Tensor containing the embeddings produced by each intermediate
            layer for the input batch.
        """
        
        emb: list = []
        for conv in self.conv_layers:
            x_res: torch.Tensor = x
            x: torch.Tensor = conv(x)
            if x.shape == x_res.shape:
                x = x + x_res
        emb.append(x)

        x = self.gelu(x)

        x_avg = F.adaptive_avg_pool1d(x, 1)
        x_max = F.adaptive_max_pool1d(x, 1)
        x = torch.cat([x_avg, x_max], dim=1).flatten(1)
        emb.append(x)

        x = self.ff_layers(x)
        emb.append(x)
        return emb


def compute_train_stats(train_loader :torch.utils.data.DataLoader)->Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute mean and standard deviation of training targets.

    Parameters
    ----------
    train_loader : DataLoader
        PyTorch DataLoader of the data

    Returns
    -------
    train_mean : torch.Tensor
        Mean of all training targets.
    train_std : torch.Tensor
        Standard deviation of all training targets.
    """
    
    all_targets: list = []

    for _, y, _, _, _ in train_loader:
        all_targets.append(y)

    all_targets: torch.Tensor = torch.cat(all_targets, dim=0)

    train_mean: torch.Tensor = all_targets.mean()
    train_std: torch.Tensor = all_targets.std().clamp(min=1e-8)
    
    return train_mean, train_std

def normalize_targets(y: torch.Tensor, train_mean: torch.Tensor, train_std: torch.Tensor)->torch.Tensor:
    """
    Normalize targets using training set statistics.
    
    Parameters
    ----------
    y : torch.Tensor
        Target tensor.
    train_mean : torch.Tensor
        Mean computed from training targets.
    train_std : torch.Tensor
        Standard deviation computed from training targets.

    Returns
    -------
    torch.Tensor
        Normalized target tensor.
    """
    
    return (y - train_mean) / train_std

def get_embeddings(model: ProtCNN, dataset: torch.utils.data.TensorDataset, batch_size: int=256)->torch.Tensor:
    """
    Extract the last layer embeddings from a dataset using a trained model.

    Parameters
    ----------
    model : ProtCNN
        Trained model that implements `extract_embedding`.
    dataset : TensorDataset
        Dataset containing the training data.
    batch_size : int, optional
        Number of samples per batch during inference.

    Returns
    -------
    torch.Tensor
        Concatenated embeddings for all samples in the dataset.
    """
    
    model.eval()
    loader:torch.utils.data.DataLoader = DataLoader(dataset, batch_size=batch_size)
    all_emb: list = []

    with torch.no_grad():
        for x, _, _, _, _ in loader:
            x = x.to(device)
            emb: torch.Tensor = model.extract_embedding_last(x)
            all_emb.append(emb.cpu())
    return torch.cat(all_emb)


def train(train_dataset: TensorDataset, val_dataset: TensorDataset, al_cfg: ActiveLearningConfig, data_cfg: DataConfig, train_cfg: TrainConfig, model_cfg: ModelConfig, 
          model: Optional[ProtCNN]=None, verbose: bool=False)->tuple[ProtCNN, torch.Tensor, torch.Tensor]:
    """
    Trains either a provided model on the training dataset or creates a new model if none is provided
    ----------
    model : Optional[CNN1D]
        Model used for training, if empty a new model is created.
    train_dataset:TensorDataset
        Used to train the model.
    val_dataset:TensorDataset
        Used to during training.
    train_cfg : TrainConfig
        Contains Paramters important for training.
    model_cfg : ModelConfig
        Contains Paramters important for the model architecture.

    Returns
    -------
    CNN1D
        The trained model.
    """

    if model is None:

        model: ProtCNN = ProtCNN(
                    feat_dim=train_dataset[0][0].shape[0],
                    hidden_channels=model_cfg.hidden_channels,
                    depth_conv=model_cfg.depth_conv,
                    depth_ffnn=model_cfg.depth_ffnn,
                    dropout_rate=model_cfg.dropout_rate
                )
        
        epochs: int = train_cfg.epochs * train_cfg.starting_epoch_multiplier
    else:
        epochs: int = train_cfg.epochs

    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    huber_loss = torch.nn.HuberLoss(delta=1.35)

    running_patience: int = 0
    best_model = copy.deepcopy(model.state_dict())
    best_val_loss: float = float("inf")
    
    num_workers: int = min(8, os.cpu_count() // 2)
    train_loader = DataLoader(train_dataset, batch_size=data_cfg.batch_size, shuffle=True,  num_workers=num_workers 
                              ,pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    val_loader   = DataLoader(val_dataset,   batch_size=data_cfg.batch_size, shuffle=False, num_workers=num_workers ,
                              pin_memory=True, persistent_workers=True, prefetch_factor=4)
    
    train_mean, train_std = compute_train_stats(train_loader)
        
    for epoch in range(epochs):
        
        model.train()
        avg_train_loss: float = 0.0
        avg_val_loss: float = 0.0
         
        for xb, yb, ob, _, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ob = ob.to(device, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                
                out_reg: torch.Tensor = model(xb)

                yb: torch.Tensor = normalize_targets(y=yb,train_mean=train_mean,train_std=train_std)
                
                mask = (ob == 1)
                weights = torch.full_like(
                    yb,
                    1.0 - epoch / epochs,
                )
                weights[mask] = al_cfg.new_query_weight 
                weights: torch.Tensor = torch.ones_like(yb)
                    
                loss_fn__no_reduc = torch.nn.HuberLoss(reduction="none")
                loss: torch.Tensor = loss_fn__no_reduc(out_reg, yb)
                    
                loss = (loss * weights).mean()
                
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            avg_train_loss += loss.detach() * xb.size(0)

        avg_train_loss: float = (avg_train_loss / len(train_loader.dataset)).item()

        if val_loader is not None:
            model.eval()

            with torch.no_grad():
                for xb, yb, _ , _, _ in val_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    
                    yb: torch.Tensor = normalize_targets(y=yb,train_mean=train_mean,train_std=train_std)
                    
                    reg_out: torch.Tensor = model(xb)
                    loss: torch.Tensor = huber_loss(reg_out,yb)
                  
                        
                    avg_val_loss += loss.detach() * xb.size(0)

            avg_val_loss: float = (avg_val_loss / len(val_loader.dataset)).item()
      
            running_patience += 1
            if avg_val_loss < best_val_loss - 1e-6:
                best_val_loss = avg_val_loss
                best_model = {k: v.cpu() for k, v in model.state_dict().items()}
                running_patience = 0

            if running_patience >= model_cfg.patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1} with best val loss {best_val_loss:.4f}", flush=True)
                break
          
        scheduler.step()    
        if verbose:
            if val_loader is not None:
                print(f"\nEpoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}", flush=True)
            else:
                print(f"\nEpoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}", flush=True)


    if val_loader is not None:
        model.load_state_dict(best_model)

    return model, train_mean, train_std

@torch.no_grad()
def predict(model: ProtCNN, test_dataset: torch.utils.data.TensorDataset, num_samples: int=20, batch_size: int=1024, y_mean: float=0.0,
            y_std: float=1.0)->Tuple[np.ndarray, np.ndarray]:
    
    model: ProtCNN = model.to(device)

    model.train()
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.eval()

    loader: DataLoader = DataLoader(test_dataset, batch_size=min(batch_size, len(test_dataset)), shuffle=False)

    all_means: list = []
    all_vars: list = []

    for xb, _, _, _, _ in loader:
        xb = xb.to(device)

        preds: list = []

        for _ in range(num_samples):
            preds.append(model(xb))

        preds: torch.Tensor = torch.stack(preds, dim=0) 

        mean: torch.Tensor = preds.mean(dim=0)
        var: torch.Tensorr = preds.var(dim=0)
        
        mean = mean * y_std + y_mean
        var = var * (y_std ** 2)

        all_means.append(mean)
        all_vars.append(var)

    mean_preds: np.ndarray = torch.cat(all_means).cpu().numpy()
    var_preds: np.ndarray = torch.cat(all_vars).cpu().numpy()

    return mean_preds, var_preds

@torch.no_grad()
def evaluate(model:ProtCNN, dataset:TensorDataset, mc_samples: int=100, y_mean: float=0.0, y_std: float=1.0)->Dict[str, float]:
    """
    Evaluates a model on the given TensorDataset.

    Parameters
    ----------
    topk_percent
    model : CNN1D
        Model for evaluation.
    dataset:TensorDataset
        Data we want the model to evaluated on.
    mc_samples : int
        Number of forward passes used in Monte Carlo Dropout Sampling.
    y_mean : float
        Mean of the target variable.
    y_std : float
        Standard deviation of the target variable.
    """
    mean, var = predict(model, dataset, num_samples=mc_samples, y_mean=y_mean, y_std=y_std)

    preds = mean.flatten()
    targets = torch.cat([y for _, y, _, _, _ in dataset]).cpu().numpy().flatten()

    r2 = r2_score(targets, preds)
    mae = mean_absolute_error(targets,preds)
    nmae = mae / np.std(targets)
    
    pearson = pearsonr(preds, targets)[0]
    spearman = spearmanr(preds, targets)[0]
    tau = kendalltau(targets, preds)[0]
    wd = wasserstein_distance(targets, preds)
   
    error = np.abs(targets - preds)
    corr, _ = spearmanr(var.flatten(), error)
        
    print(
        f"Eval (All) | "
        f"R²: {r2:.3f} | "
        f"Mae: {mae:.3f} | "
        f"NMae: {nmae:.3f} | "
        f"Pearson: {pearson:.3f} | "
        f"Spearman: {spearman:.3f} | "
        f"Kendalltau: {tau:.3f} | "
        f"Wasserstein Distance: {wd:.3f} |"
        f"Error_var_corr: {corr:.3f} |"
    )

    return {
        "R2": r2,
        "Pearson": pearson,
        "Spearman": spearman,
        "Kendalltau": tau,
    }