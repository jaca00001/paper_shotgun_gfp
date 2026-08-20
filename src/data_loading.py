import os
import torch
import numpy  as np
import pandas as pd

from torch.utils.data        import TensorDataset
from typing                  import Tuple, Union, Optional, List

from src.configs             import DataConfig
from sklearn.model_selection import train_test_split


def prepare_data_protein(filepath: str,  rescale: str="none", sequence_row_label: str="sequence", target_row_label: str="fitness")->pd.DataFrame:
    """
        Read a CSV/TSV file and load protein sequences and target values into a pandas DataFrame.

        Parameters
        ----------
        filepath : str
            Path to the input file.
        sequence_row_label : str
            Name of the sequence column.
        target_row_label : str
            Name of the target column.
        rescale : bool
            Whether to apply rescaling scaling to the target values.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing the sequences and target values.
    """
    
    file_name: str = os.path.basename(filepath)
    
    df: pd.DataFrame = pd.read_csv(filepath, sep="," if filepath.endswith(".csv") else "\t")
    
    df["gene"] = file_name[:-4]
    df = df.rename(columns={sequence_row_label: "sequence"})
    df = df.rename(columns={target_row_label: "fitness"})
    
    if rescale == "unscale":
        df["fitness"] = 10 ** df["fitness"]
    elif rescale == "log10":
        df["fitness"] = np.log10(df["fitness"])
    
    return df

def prepare_data_multiple_proteins(folderpath: str, sequence_row_label: str="sequence", target_row_label: str="fitness", rescale: bool=True,
                                   concat_peaks: bool=False, additional_peaks: List[pd.DataFrame]=None)->Union[List[pd.DataFrame], pd.DataFrame]:
    """
        Read a in a folder of CSV/TSV files and loads the protein sequences and target values into a list of pandas DataFrames.

        Parameters
        ----------
        folderpath : str
            Path to the input folder.
        sequence_row_label : str
            Name of the sequence column.
        target_row_label : str
            Name of the target column.
        rescale : bool
            Whether to apply log10 scaling to the target values.
        concat: bool
            Wheter to concat the list of pd.DataFrames into a single one.
        additional_peaks: List[pd.DataFrame]
            Additional dataframes can be appended that are stored in a diffefrent directory.

        Returns
        -------
        list[pandas.DataFrame] or pandas.DataFrame
            If concat is False, returns a list of DataFrames.
            If concat is True, returns a single concatenated DataFrame.
    """
    dfs: List[pd.DataFrame] = []

    for i, filename in enumerate(os.listdir(folderpath)):
        filepath: str = os.path.join(folderpath, filename)
        df_tmp: pd.DataFrame = prepare_data_protein(filepath, sequence_row_label, target_row_label, rescale)
        df_tmp["source_id"] = i
        dfs.append(df_tmp)

    
    if additional_peaks is not None:
        for j in range(len(additional_peaks)):
            additional_peaks[j]["source_id"] = i + j + 1
        
            dfs.extend(additional_peaks)
    
    return dfs if not concat_peaks else pd.concat(dfs, ignore_index=True)

def split_data(train_df: pd.DataFrame, data_cfg: DataConfig, unlabeled_df: Optional[pd.DataFrame]=None, random: bool=True, 
               split: str = "perc")->Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
        Splits the data either by percentages ("perc") or by mutation count ("mut").

        Parameters
        ----------
        train_df : pd.DataFrame
            Labeled dataset.
        unlabeled_df : Optional[pd.DataFrame]
            Optional unlabeled dataset.
        random : bool
            If False, uses a fixed random seed.
        split : str
            Either "perc" or "mut" or "both".

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            train, validation and unlabeled datasets.
    """

    random_state: int = None if random else 42

    if split == "perc":
        val_size: float = data_cfg.val_size - 1e-6

        if unlabeled_df is None or len(unlabeled_df) == 0:
            total_used: float = data_cfg.train_size + val_size + data_cfg.test_size
            
            train_frac: float = data_cfg.train_size / total_used
            val_frac: float = val_size / total_used
            test_frac: float = data_cfg.test_size / total_used

            used_df, _ = train_test_split(train_df, train_size=total_used, random_state=random_state)

            train_df, remaining_df = train_test_split(used_df, train_size=train_frac, random_state=random_state)

            val_frac_remaining: float = val_frac / (val_frac + test_frac)

            val_df, unlabeled_df = train_test_split(remaining_df, train_size=val_frac_remaining, random_state=random_state)

        else:
            total_used: float = data_cfg.train_size + val_size

            used_df, _ = train_test_split(train_df, train_size=total_used, random_state=random_state)

            train_df, val_df = train_test_split(used_df, train_size=data_cfg.train_size / total_used, random_state=random_state)

    elif split == "mut":
        low_mut_df: pd.DataFrame = train_df[train_df["num_mutations"] <= 3]
        
        if unlabeled_df is None or len(unlabeled_df) == 0:
            unlabeled_df: pd.DataFrame = train_df[train_df["num_mutations"] >= 4]
        else: 
            unlabeled_df: pd.DataFrame = unlabeled_df[unlabeled_df["num_mutations"] >= 4]

        train_df, val_df = train_test_split(low_mut_df, train_size=0.9, random_state=random_state)

    
    elif split == "both":
        val_size: float = data_cfg.val_size - 1e-6
        low_mut_df: pd.DataFrame = train_df[train_df["num_mutations"] <= 3]
        
        if unlabeled_df is None or len(unlabeled_df) == 0:
            unlabeled_df: pd.DataFrame = train_df[train_df["num_mutations"] >= 4]
        else: 
            unlabeled_df: pd.DataFrame = unlabeled_df[unlabeled_df["num_mutations"] >= 4]
        
        total_used: float = data_cfg.train_size + val_size

        low_mut_df_used, _ = train_test_split(low_mut_df, train_size=total_used, random_state=random_state)
        
        train_df, val_df = train_test_split(low_mut_df_used, train_size=data_cfg.train_size / total_used, random_state=random_state)
       
    else:
        raise ValueError(f"Unknown split mode '{split}'. Use 'perc' or 'mut'.")

    train_df: pd.DataFrame = train_df.copy()
    val_df: pd.DataFrame = val_df.copy()
    unlabeled_df: pd.DataFrame = unlabeled_df.copy()

    train_df["origin"] = "train"
    val_df["origin"] = "train"
    unlabeled_df["origin"] = "unlabeled"

    return train_df, val_df, unlabeled_df

def separate_peaks(peaks_df: pd.DataFrame, train_peaks: list[pd.DataFrame], test_peaks: list[pd.DataFrame]=None)->Tuple[pd.DataFrame,pd.DataFrame]:
    """
        Merge peak DataFrames according to the provided train/test genes.

        Parameters
        ----------
        peaks_df : list[pd.DataFrame]
            List of peak DataFrames.
        train_indices : list[int]
            Indices of DataFrames to include in the training set.
        test_indices : list[int] | None, optional
            Indices of DataFrames to include in the test set. If None, empty,
            or identical to train_indices, no test DataFrame is returned.

        Returns
        -------
        tuple[pd.DataFrame, pd.DataFrame | None]
            Merged training DataFrame and optionally the merged test DataFrame.
    """
    
    train_indices: list[str] = [i for i, df in enumerate(peaks_df) if df["gene"].iloc[0] in train_peaks]
    test_indices: list[str] = [i for i, df in enumerate(peaks_df) if df["gene"].iloc[0] in test_peaks] if test_peaks is not None and test_peaks != [] else []

    train_set: set[int] = set(train_indices)
    train_df: pd.DataFrame  = pd.concat(df for i, df in enumerate(peaks_df) if i in train_set)

    test_df: pd.DataFrame = pd.DataFrame()
    if test_indices and test_indices != train_indices:
        test_set: set[int] = set(test_indices)
        test_df = pd.concat(df for i, df in enumerate(peaks_df) if i in test_set)

    return train_df, test_df

def onehot_encode_sequences(sequences: List[str])->np.ndarray:
    """
        Returns a numpy array of one-hot encoded sequences.

        Parameters
        ----------
        sequences : list[str]
            List of the sequences which should be encoded

        Returns
        -------
        np.ndarray
            Returns a numpy array containing the encoded sequences.
    """
    amino_acids: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
    
    seq_len: int = len(sequences[0])
    num_aa: int = len(amino_acids)
    aa_to_idx: dict[str, int] = {aa: i for i, aa in enumerate(amino_acids)}

    encoded: np.array = np.zeros((len(sequences), seq_len, num_aa), dtype=np.float32)
    for i, seq in enumerate(sequences):
        for j, aa in enumerate(seq):
            if aa in aa_to_idx:
                encoded[i, j, aa_to_idx[aa]] = 1.0

    return encoded.transpose(0, 2, 1)

def create_dataset(df: pd.DataFrame, origin: Optional[List[str]]=None)->TensorDataset:
    """
        Turns a pd.Dataframe Object into a TensorDataset with encoded sequences.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe which should be turned into the Dataset
        origin: Optional[list[int]]
            Optional and can be ignored, used to mix loss fucntions for points already labeled (origin=1) or points added during active learning (origin=0).

        Returns
        -------
        TensorDataset
            TensorDataset of the DataFrame containing encoded sequences, target and origin.
    """
    
    sequences: list[str] = df["sequence"].tolist()
    
    X: torch.Tensor = torch.as_tensor(onehot_encode_sequences(sequences), dtype=torch.float32)
    
    y: torch.Tensor = torch.as_tensor(df["fitness"].to_numpy(), dtype=torch.float32).unsqueeze(1)

    if origin is None:
        o: torch.Tensor = torch.zeros(len(df), dtype=torch.int32)
    else:
        o: torch.Tensor = torch.as_tensor(np.asarray(origin) != "train", dtype=torch.int32)
        
    source: torch.Tensor = (torch.as_tensor(df["source_id"].to_numpy(),dtype=torch.int32)
        if "source_id" in df.columns else                
                            torch.zeros(len(df), dtype=torch.int32))
    
    epistatic: torch.Tensor = torch.zeros(len(df), dtype=torch.int32)
    
    return TensorDataset(X, y, o, source, epistatic)