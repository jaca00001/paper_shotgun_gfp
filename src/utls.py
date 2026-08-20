
import gc
import torch
import random
import optuna
import numpy  as np
import pandas as pd

from optuna.samplers  import TPESampler
from collections      import defaultdict
from optuna.pruners   import HyperbandPruner
from sklearn.cluster  import SpectralClustering
from typing           import Optional, Set, Tuple, List
from torch.utils.data import TensorDataset, Subset, ConcatDataset

from src.model        import *
from src.configs      import *
from src.data_loading import *
from src.ploting      import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup()->None:
    """
        Sets the backend variables.

        Parameters
        ----------

        Returns
        -------
        
    """
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.autograd.set_detect_anomaly(False)
    torch.autograd.profiler.profile(False)

def create_config(**kwargs):
    """
        Creates and returns the config with all paramters for training.

        Parameters
        ----------
        data_cfg: Dataset configuration.
            Expected keys:
                - train_size (float): Fraction of data used for training.
                - val_size (float): Fraction of data used for validation.
                - test_size (float): Fraction of data used for testing.
                - batch_size (int): Batchsize of the training and validation set.

        train_cfg: Training configuration.
            Expected keys:
                - use_al (bool): Whether to use active learning.
                - num_runs (int): Number of independent training runs.
                - epochs (int): Number of epochs the model is trained for.
                - starting_epoch_multiplier (int): Normal training or the first round of active learning trains for starting_epoch_multiplier * epochs
                - lr (float): The learning rate of the optimizer.
                - weight_decay (float): The weight decay rate of the optimizer.
        
        
        model_cfg: Model configuration.
            Expected keys:
                - depth_conv (int): Number of convolutional layers.
                - depth_ffnn (int): Number of fully connected layers.
                - hidden_channels (int): Initial number of hidden_channels.
                - dropout_rate (float): Dropout rate of the fully connected layers.
                - patience (int): After patience epochs without imprvoment on the validions set is the training stopped and the best model chosen.
        

        al_cfg: Active learning configuration.
            Expected keys:
                - acquisition_budget_total (int): Total number of samples to acquire.
                - rounds (int): Number of active learning rounds.
                - new_query_weight (float): Loss upscaling for newly qurried points.
                - alpha (float): Weight parameter for the acquistion function.

        Returns
        -------
        Config
            A Config object containing the merged configuration
    """
    
    data_cfg: DataConfig = DataConfig(**kwargs.get("data_cfg", {}))
    train_cfg: TrainConfig = TrainConfig(**kwargs.get("train_cfg", {}))
    model_cfg: ModelConfig = ModelConfig(**kwargs.get("model_cfg", {}))
    al_cfg: ActiveLearningConfig = ActiveLearningConfig(**kwargs.get("al_cfg", {}))

    if not train_cfg.use_al:
        for k, v in NON_AL_TRAIN_DEFAULTS.items():
            setattr(train_cfg, k, v)
            
      
        for k, v in NON_AL_MODEL_DEFAULTS.items():
            setattr(model_cfg, k, v)

    return Config(
        data_cfg=data_cfg,
        train_cfg=train_cfg,
        model_cfg=model_cfg,
        al_cfg=al_cfg,
    )

def assign_id_to_gene(peaks_df: List[pd.DataFrame]):
    """
    Assigns each gene a unique id.

    Parameters
    peaks_df: List[pd.DataFrame]: Dataframes containing the gene names as a colummn.

    Returns
    -------
    dict[str, str]
        A mapping from gene to id.
    """
    
    id_to_source: dict[str, str] = {}
    
    for p in peaks_df:
        id_to_source[str(p["source_id"].iloc[0])] = p["gene"].iloc[0]
    
    return id_to_source
    
def get_peaks_sources(peaks_df: List[pd.DataFrame], test_peak: List[str]=[])->Tuple[List[str],dict[str, str]]:
    """
        Split the Data into wt peak, shotgun peaks as and test peak. Also returns a mapping from gene to a unique id.

        Parameters
        peaks_df: List[pd.DataFrame]: Dataframes containing the gene data.
        test_peak: List[str]: List containing the genes which should be excluded from the training set.

        Returns
        -------
        List[str]
            List of all gene names.
        dict[str, str]
            A mapping from id to gene. 
    """

    gene_to_id = assign_id_to_gene(peaks_df)
   
    multiple_df = [df["gene"].iloc[0] for df in peaks_df if df["gene"].iloc[0] not in test_peak]
    
    return  multiple_df, gene_to_id

def get_mind_dists(model: ProtCNN, labeled_dataset: Subset, unlabeled_dataset: Subset, chunk_size: int=1024)->tuple[np.ndarray,torch.Tensor]:
    """
        Returns the min distance of the unlabaled embeddings to the already labeled embeddings.
    
        Parameters
        ----------
        model : ProtCNN
            Model used in training and to compute the embeddings
        labeled_dataset : torch.utils.data.Subset
            Dataset containing the labeled examples.
        unlabeled_dataset : torch.utils.data.Subset
            Dataset containing the unlabeled examples.
        chunk_size : int
            Batch size to control memory usage.

        Returns
        -------
        np.ndarray
            Min distances.
        torch.Tensor
            The unlabeled embeddings.
    """

    labeled_emb: torch.Tensor = get_embeddings(model, labeled_dataset)
    unlabeled_emb: torch.Tensor = get_embeddings(model, unlabeled_dataset)
 
    min_dists = []

    for i in range(0, unlabeled_emb.size(0), chunk_size):
        chunk: torch.Tensor = unlabeled_emb[i:i+chunk_size]
        dist: float = torch.cdist(chunk, labeled_emb)
        min_chunk, _ = torch.min(dist, dim=1)
        min_dists.append(min_chunk)

    min_dist: np.array = torch.cat(min_dists).numpy()
    return min_dist, unlabeled_emb

def kmers(seq: str, kmer_size: int=3)->Set[str]:
    """
        Returns a set of k-mers of the given sequence.

        Parameters
        ----------
        seq : str
            Amino acid sequence.
        kmer_size : int
            Size of the k-mers.

        Returns
        -------
        set[str]
            Set containing the k-mers of the sequence.
    """
    return {seq[i:i+kmer_size] for i in range(len(seq) - kmer_size + 1)}

def compute_SE(metrics_list)->None:
    """
        Computes the standard error for all metrics.

        Parameters
        ----------
        metrics_list : list(dict)
            List of the different results for the same experiment.

        Returns
        -------
    """
    
    if len(metrics_list) > 1:
        aggregated = defaultdict(list)

        for m in metrics_list:
            for key, value in m.items():
                aggregated[key].append(value)

        for metric in aggregated:
            values = aggregated[metric]
            se = np.std(values) / np.sqrt(len(values))
            print(f"Metric: {metric} Mean: {np.mean(values):.4f} ± {se:.4f} (n={len(values)})")

def update_state(state: ALState, new_samples: List[int], pool_dataset: TensorDataset, val_dataset: Subset)-> tuple[ALState,ConcatDataset]:
    """
        Moves selected samples from unlabeled -> labeled.
        Adds a number of new samples to validation set.
    
        Parameters
        ----------
        state : ALState
            List of indices of labeled and unlabeled points
        new_samples : List[int]
            List of newly sampled indices
        pool_dataset : TensorDataset
            Combined dataset
        val_dataset : TensorDataset
            Validation Set

        Returns
        -------
        ALState
            Updated indices.
        TensorDataset
            Updated validation set.
    """

    random.shuffle(new_samples)
    new_set: set[int] = set(new_samples)
    
    n_val = max(1, int(0.05 * len(new_samples)))

    val_samples: list[int] = new_samples[:n_val]
    train_samples: list[int] = new_samples[n_val:]

    updated_labeled: list[int] = state.labeled + train_samples

    updated_unlabeled: list[int] = [i for i in state.unlabeled if i not in new_set]

    val_subset = Subset(pool_dataset, val_samples)
    updated_val_dataset = ConcatDataset([val_dataset,val_subset])

    return (
        ALState(
            labeled=updated_labeled,
            unlabeled=updated_unlabeled,
        ),
        updated_val_dataset
    )

def eval_round(model: ProtCNN, val_dataset: Subset, unlabeled_dataset: Subset, 
               exp_name:str, r:int, trial:Optional[optuna.trial.Trial]=None, y_mean:float=None, y_std:float=None)->float:
    
    """
        Evaluates the current model using the unlabeled set, if enabled and hpo is running can be pruned using the validation dataset. 
    
        Parameters
        ----------
        model : ProtCNN
            Model used in training and to compute the embeddings
        val_dataset : torch.utils.data.Subset
            Dataset containing the validation examples.
        unlabeled_dataset : torch.utils.data.Subset
            Dataset containing the unlabeled examples.
        exp_name : str
            Name of the folder the results will be saved in.
        trial: Optional[optuna.trial.Trial]
            Can be ignored, is used to prune the trial when hpo is used

        Returns
        -------
        float
            Current metric of the model.
        
    """
    
    
    if trial is not None:
   
        metrics: float = evaluate(model, val_dataset, mc_samples=15)["R2"]
        trial.report(metrics, step=r)

        if trial.should_prune():
            print(f"Trial pruned at AL round {r}")
            raise optuna.exceptions.TrialPruned()
        
        metrics: dict = save_metrics(model=model,unlabeled_dataset=unlabeled_dataset,id_plot=r,location=exp_name,y_mean=y_mean,y_std=y_std)
   
    else:
        metrics: dict  = save_metrics(model=model,unlabeled_dataset=unlabeled_dataset,id_plot=r,location=exp_name, y_mean=y_mean, y_std=y_std
        )
    return metrics

def get_budget(al_cfg: ActiveLearningConfig, r: int, unlabeled_df:pd.DataFrame, uniform:bool=False)->int:
    
    """
        Returns the budget of the current AL round based on the strategy
    
        Parameters
        ----------
        al_cfg : ActiveLearningConfig
            Config containing parameters regarding AL
        r : int
            Current active learning round.
        unlabeled_df : pd.DataFrame
            Dataset containing the unlabeled examples.
        uniform : bool
            Wheter to sample uniformly
        

        Returns
        -------
        int
            Current budget, at least one.
        
    """
    
    
    if uniform:
        budget_total: float = al_cfg.acquisition_budget_total * len(unlabeled_df) if al_cfg.acquisition_budget_total <= 1 else al_cfg.acquisition_budget_total
        budget_per_round: int = int(budget_total/ (al_cfg.rounds)) if al_cfg.rounds > 0 else 0
    else:
        harmonic: float = np.sum([1 / n for n in range(1, al_cfg.rounds + 1)])
        acquisition_budget_start: float = al_cfg.acquisition_budget_total / harmonic
        budget_per_round: int = int(acquisition_budget_start / (r + 1))

    return  max(1, budget_per_round)

def get_experiment_name(cfg: Config, train_peaks: list[str], test_peaks: list[str], round: int)->str:
    
    """
        Automatically names the folder based on the data and the config.

        Parameterss
        ----------
        cfg : pd.DataFrame
            Dataset used for training.
        train_idx: list[int]
            Genes used for training
        test_idx: list[int]
            Genes used for testing
        unlabeled_df: pd.DataFrame
            Dataset containing the unlabeled examples.
        round: int
            Curent round of experiment when performing multiple runs of the same experiment.
    
        Returns
            -------
        str
            Folder name for the experiment results.
    """
   

    exp_name: str = "AL" if cfg.train_cfg.use_al else "FT_" 
    
    train_idx: str = "_".join(idx for idx in train_peaks)   
    test_idx: str = "_".join(idx for idx in test_peaks) if test_peaks is not None and test_peaks != [] else train_idx
    
    exp_train: str = "_train_" + train_idx if len(train_idx) < 20 else "Multiple_Peaks" + "..." if len(train_idx) > 15 else ""
    exp_test: str =  "_test_" + test_idx if len(test_idx) < 20 else "Multiple_Peaks" + "..." if len(test_idx) > 15 else ""
        
    if round == 0:
        print(f"Training: {exp_train}, Testing: {exp_test}")

    exp_name += exp_train + exp_test
    exp_name += f"_train_s_{cfg.data_cfg.train_size}_val_s_{cfg.data_cfg.val_size}_test_s_{cfg.data_cfg.test_size}"
    exp_name += f"_q_{cfg.al_cfg.acquisition_budget_total  if cfg.train_cfg.use_al else 0}" 
    
    return exp_name + f"_{cfg.data_cfg.split}"

def select_distance_based(seqs: List[str], k: int, kmer_size: int=3)->List[int]:
    """
        Given a list of n sequences selects k spread out sequences and returns the indices.

        Parameters
        ----------
        seqs : list[str]
            Amino acid sequences.
        k : int
            Number of sequences to return.
        kmer_size : int
            Size of the k-mers the sequences are split into to compare similarity.

        Returns
        -------
        list[int]
            List of selected indices.
    """
    n: int = len(seqs)
    L: int = len(seqs[0])
    num_kmers: int = L - kmer_size + 1

    kmer_sets: list[Set[str]] = [kmers(s, kmer_size) for s in seqs]

    selected: list[int] = []
    selected_mask: np.array = np.zeros(n, dtype=bool)

    selected.append(0)
    selected_mask[0] = True

    min_dist: np.array = np.full(n, np.inf)

    while len(selected) < k:
        last_selected: str = selected[-1]
        last_kmer_set: Set[str] = kmer_sets[last_selected]

        for i in range(n):
            if selected_mask[i]:
                continue

            inter: int = len(last_kmer_set & kmer_sets[i])
            d: float = 1.0 - inter / (2*num_kmers - inter)

            if d < min_dist[i]:
                min_dist[i] = d

        next_idx: int = np.argmax(
            np.where(selected_mask, -1.0, min_dist)
        )

        selected.append(next_idx)
        selected_mask[next_idx] = True

    return selected

def compute_acquisition(model: ProtCNN, labeled_dataset: TensorDataset, unlabeled_dataset: TensorDataset, al_cfg: ActiveLearningConfig)->tuple[np.ndarray,torch.Tensor]:
      
    """
        Computes the acquisition scores.
    
        Parameters
        ----------
        model : ProtCNN
            Model used in training and to compute the embeddings
        al_cfg: ActiveLearningConfig
            Config containing the AL parameters.

        labeled_dataset: torch.utils.data.Subset
            Subset of the data which is labeled.
        unlabeled_dataset:torch.utils.data.Subset
            Subset of the data which is unlabeled.
        Returns
        -------
        np.ndarray
            acquisition scores
        torch.Tensor
            Embeddings of the unlabeled data.
    """
    
    
    min_dist, unlabeled_emb = get_mind_dists(model, labeled_dataset, unlabeled_dataset) 
    mean, uncert = predict(model, unlabeled_dataset, num_samples=25)  
    
    uncert = (uncert - uncert.min()) / (uncert.max() - uncert.min() + 1e-8)
    min_dist = (min_dist - min_dist.min()) / (min_dist.max() - min_dist.min() + 1e-8)
    mean = (mean - mean.min()) / (mean.max() - mean.min() + 1e-8)
    
    min_dist: np.ndarray = np.asarray(min_dist).squeeze()
    uncert: np.ndarray = np.asarray(uncert).squeeze()
    mean: np.ndarray = np.asarray(mean).squeeze()
    
    acquisition: np.ndarray = al_cfg.alpha * uncert + (1 - al_cfg.alpha) * min_dist 
    return acquisition, unlabeled_emb

def compute_assignments(unique_clusters: np.ndarray, assignments: np.ndarray, unlabeled_global: np.ndarray, budget: int)->list[int]:
    
    """
    Computes the acquisition score.
   
    Parameters
    ----------
    unique_clusters: np.ndarray
        List containing the unique cluster ids.
    assignments: np.ndarray
        cluster assginment of each point
    budget: int
        Total Budget to distribute
    Returns
    -------
    list[int]
       Distribtuion of the budget for each cluster.
    """
    
    cluster_budgets: list = []
    fractions: list = []

    non_empty_clusters: list = []

    for cluster_id in unique_clusters:
        selected_unlabeled_idx: np.ndarray = np.where(assignments == cluster_id)[0]

        if len(selected_unlabeled_idx) == 0:
            cluster_budgets.append(0)
            fractions.append(0)
        else:
            cluster_budgets.append(1)
            fractions.append(0)
            non_empty_clusters.append(cluster_id)

    remaining_budget: int = budget - sum(cluster_budgets)

    if remaining_budget > 0:
        for i, cluster_id in enumerate(unique_clusters):
            selected_unlabeled_idx = np.where(assignments == cluster_id)[0]

            if len(selected_unlabeled_idx) == 0:
                continue

            exact_extra = (len(selected_unlabeled_idx) / len(unlabeled_global)) * remaining_budget

            extra_floor = int(np.floor(exact_extra))
            cluster_budgets[i] += extra_floor
            fractions[i] = exact_extra - extra_floor

        leftover: int = budget - sum(cluster_budgets)

        if leftover > 0:
            order = np.argsort(fractions)[::-1]
            for idx in order:
                if leftover == 0:
                    break
                if cluster_budgets[idx] > 0:  
                    cluster_budgets[idx] += 1
                    leftover -= 1
        
    return cluster_budgets

def get_queried_samples(model: ProtCNN, al_cfg: ActiveLearningConfig, budget: int, state: ALState, base_dataset: list, 
                        labeled_dataset: torch.utils.data.Subset, unlabeled_dataset: torch.utils.data.Subset)->List[int]:
    
    """
        Computes the acquisition score for each point and returns the next points to be querried.
    
        Parameters
        ----------
        model : ProtCNN
            Model used in training and to compute the embeddings
        al_cfg: ActiveLearningConfig
            Config containing the AL parameters.
        budget: int
            Number of points to be sampled in the current round.
        state: ALState
            State of the AL process, containing the idx of the labeled and unlabeled sequences
        base_dataset: list
            Dataset containing every datapoint
        labeled_dataset: torch.utils.data.Subset
            Subset of the data which is labeled.
        unlabeled_dataset:torch.utils.data.Subset
            Subset of the data which is unlabeled.
        Returns
        -------
        list[int]
            Indices of the new selected samples.
    """
    acquisition, unlabeled_emb = compute_acquisition(model, labeled_dataset, unlabeled_dataset, al_cfg)
    
    clustering = SpectralClustering(n_clusters=2, random_state=0)
    assignments: np.ndarray = clustering.fit_predict(unlabeled_emb.cpu().numpy())
    
    unlabeled_global: np.ndarray = np.asarray(state.unlabeled)
    unique_clusters: np.ndarray = np.unique(assignments)
    selected: list = []
  
    cluster_budgets: list[int] = compute_assignments(unique_clusters, assignments, unlabeled_global, budget)
    
    for cluster_id in unique_clusters:
        selected_unlabeled_idx: np.ndarray = np.where(assignments == cluster_id)[0]
        
        if len(selected_unlabeled_idx) == 0:
            continue
        
        acquisition_cluster: np.ndarray = acquisition[selected_unlabeled_idx]
        cluster_global: np.ndarray = unlabeled_global[selected_unlabeled_idx]
        
        cluster_idx: np.ndarray = np.where(unique_clusters == cluster_id)[0][0]
        cluster_budget: int = cluster_budgets[cluster_idx]
        cluster_budget = min(len(selected_unlabeled_idx), cluster_budget)
        
        diversity_budget: int =  min(len(acquisition_cluster),cluster_budget * 5)
        top_local: np.ndarray = np.argsort(-acquisition_cluster)[:diversity_budget]
   
        top_global: list = cluster_global[top_local].tolist()
            
        seqs: list = [base_dataset[i][0] for i in top_global]
        
        diverse_local: list = select_distance_based(seqs, cluster_budget)
        
        selected.extend([top_global[i] for i in diverse_local])

    return selected

def standard_full_train(model: ProtCNN, cfg: Config, exp_name: str, train_df: pd.DataFrame, val_df: pd.DataFrame, unlabeled_df: pd.DataFrame)->Tuple[ProtCNN,list[dict]]:
    """
        Trains the given model using the provided data.
       
        Parameters
        ----------
        model : ProtCNN
            Model used in training and to compute the embeddings
        cfg: Config
            Config containing all parameters.
        exp_name: str
            Folder name for the experiment.
        train_df: pd.DataFrame
            Dataframe containing the training data.
        val_df: pd.DataFrame
            Dataframe containing the validation data.
        unlabeled_df: pd.DataFrame
            Dataframe containing the testing data.
        Returns
        -------
        ProtCNN
            Trained model.
        list[dict]
            Performance metrics of the model on the test set.
    """
    
    model, y_mean, y_std = train(model=model, train_dataset=create_dataset(train_df), val_dataset=create_dataset(val_df), data_cfg=cfg.data_cfg, al_cfg=cfg.al_cfg, train_cfg=cfg.train_cfg, 
                                         model_cfg=cfg.model_cfg, verbose=True)
            
    metric = save_metrics(model=model, unlabeled_dataset=create_dataset(unlabeled_df), id_plot="0", location=exp_name, y_mean=y_mean, y_std=y_std)
    return model, metric

def active_learning(train_df: pd.DataFrame, val_df: pd.DataFrame, unlabeled_df: pd.DataFrame, cfg, model: Optional[ProtCNN]=None, exp_name: str="al",
                    trial: optuna.trial.Trial=None)->ProtCNN:
    """
        Trains a model using actvie learning and samples from the unllabeled distributiopn we are tyring to prdict

        Parameters
        ----------
        exp_name
        train_df : pd.DataFrame
            Dataset used for training.
        val_df : pd.DataFrame 
            Dataset used for validation.
        unlabeled_df : pd.DataFrame 
            Dataset containing the unlabeled examples.
        al_cfg:ActiveLearningConfig
            cfg
        model_cfg:ModelConfig
            cfg
        train_cfg:TrainConfig
            cfg
        model : Optional[CNN1D]
            If no model is provided a new one will be created.
        Returns
            -------
        CNN1D
            The trained model
    """
   
    base_dataset:TensorDataset = create_dataset(
        pd.concat([train_df, unlabeled_df], ignore_index=True),
        origin=(train_df["origin"].tolist() + unlabeled_df["origin"].tolist()),
    )
    
    val_dataset: TensorDataset = create_dataset(val_df)
    
    n_labeled:int = len(train_df)
    n_total:int = len(base_dataset)
    
    
    print(f"Starting Active Learning with {n_labeled} labeled samples and {n_total - n_labeled} unlabeled samples.")
    
    state = ALState(
        labeled=list(range(n_labeled)),
        unlabeled=list(range(n_labeled, n_total))
    )
    
    for r in range(0, cfg.al_cfg.rounds + 1):
        budget = get_budget(cfg.al_cfg, r, unlabeled_df, uniform=True) 
        print(f"Active Learning Round {r}/{cfg.al_cfg.rounds} with a budget of {budget} samples.")

        labeled_dataset = Subset(base_dataset, state.labeled)
        unlabeled_dataset = Subset(base_dataset, state.unlabeled)
        
        
        model, y_mean, y_std = train(model=model,train_dataset=labeled_dataset,val_dataset=val_dataset,al_cfg=cfg.al_cfg,
                               train_cfg=cfg.train_cfg,model_cfg=cfg.model_cfg, data_cfg=cfg.data_cfg, verbose=True)
        
        if r < cfg.al_cfg.rounds:
          
            new_samples = get_queried_samples(model, cfg.al_cfg, budget, state, base_dataset, labeled_dataset, unlabeled_dataset)
            state, val_dataset = update_state(state, new_samples, base_dataset, val_dataset)
        
        unlabeled_dataset = Subset(base_dataset, state.unlabeled)
        metrics = eval_round(model, val_dataset, unlabeled_dataset, exp_name=exp_name, r=r, trial=trial, y_mean=y_mean, y_std=y_std)
            
    return model, metrics

def run_training(peaks_df: list[pd.DataFrame], train_peaks: List[str], test_peaks: Optional[List[str]], cfg: Config, 
                 trial: Optional[optuna.trial.Trial]=None, split: str="perc") -> Tuple[ProtCNN,float]:
    """
    Trains a model using on the train peaks and evaluates the performance on the test peaks.
    Parameters
    ----------
    exp_name
    peaks_df: list[pd.DataFrame]
        List of dataframes of all the genes.
    train_peaks: List[str]
        Name of the genses used for training
    test_peaks: Optional[List[str]]
        Name of the genses used for testing / AL
    cfg: Config,
        Config containing all the paratmets of the experiment
    trial:Optional[optuna.trial.Trial]
        Only active when performing hpo else None
    split: str
        Whether to train using a random split (perc), a mutation based split (mut) or both.
    Returns
        -------
    ProtCNN
        Trained model.
    float
        Metric of the model computed on the test peaks.
    """
  
    metrics_list: list[float] = []
    
    for j in range(cfg.train_cfg.num_runs):
        print(f"Run {j+1}/{cfg.train_cfg.num_runs}")
        exp_name:str = get_experiment_name(cfg, train_peaks, test_peaks, round=j) 

        train_df_merged, test_df_merged = separate_peaks(peaks_df, train_peaks, test_peaks)
        
        train_df, val_df, unlabeled_df = split_data(train_df=train_df_merged, unlabeled_df=test_df_merged, data_cfg=cfg.data_cfg)
        
        print(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Unlabeled: {len(unlabeled_df)}")
                
        if cfg.train_cfg.use_al:
            model, metric = active_learning(train_df=train_df, val_df=val_df, unlabeled_df=unlabeled_df, model=None, cfg=cfg, exp_name=exp_name,trial=trial)
        else:
            model, metric = standard_full_train(train_df=train_df, val_df=val_df, unlabeled_df=unlabeled_df, 
                                                model=None, cfg=cfg,exp_name=exp_name)
          
        metrics_list.append(metric)
    
    compute_SE(metrics_list)

    return model, metrics_list[-1]
 
def eval_model(base_path: str, cfg: Config)->None:
    """
        Evaluated a saved model using the corresponding test data.
        Parameters
        ----------
        base_path: str
            Path to the saved model and test data.
        cfg: Config,
            Config containing all the paratmets of the experiment
        
        Returns
        -------
    """
    unlab_data = torch.load(base_path +  "unlabeled_dataset.pth", map_location="cuda",  weights_only=False)

    model = ProtCNN(
                feat_dim=unlab_data[0][0].shape[0],
                hidden_channels=  cfg.model_cfg.hidden_channels,
                depth_conv=  cfg.model_cfg.depth_conv,
                depth_ffnn=  cfg.model_cfg.depth_ffnn,
                dropout_rate=  cfg.model_cfg.dropout_rate
    )
    model.load_state_dict(torch.load(base_path +  "model.pth", map_location="cuda", weights_only=False))

    mean, std = torch.load(base_path + "mean_std.pth", map_location="cuda", weights_only=False)
        
    evaluate(model,unlab_data, y_mean=mean, y_std=std)

def evaluate_all_peaks(peaks_df: list[pd.DataFrame], cfg: Config, trial: optuna.trial.Trial, metric:str="R2")->float:
    """
        Trains multiple model using every dataset in peaks_df once as the test set and returns the mean perofrmance of the model for the given metric.
        Parameters
        ----------
        peaks_df: list[pd.DataFrame]
            List containing the different datasets.
        cfg: Config,
            Config containing all the paratmets of the experiment
        trial: optuna.trial.Trial
            Current optuna trial
        metric:str
            The metric which should be computed for the model.
            
        
        Returns
        -------
        float
            Average metric over all models.
    """
    
    
    scores = []

    peak_names = [peak["gene"].iloc[0] for peak in peaks_df[:-1]]

    for test_peak in peak_names:
        train_peaks = [p for p in peak_names if p != test_peak]

        metrics = run_training(
            peaks_df=peaks_df,
            train_peaks=train_peaks,
            test_peaks=[test_peak],
            cfg=cfg,  
            trial=trial
        )

        scores.append(metrics[metric])   

    return np.mean(scores)

def run_hyperopt(peaks_df: list[pd.DataFrame], train_peaks: list[str], test_peaks: list[str], data_cfg: DataConfig,
                 n_trials: int=5000, study_name: str="hyperopt_study", use_al: bool=False, metric_type: str="R2", eval_mult: bool=False)->dict:

    """
    Trains a model using on the train peaks and evaluates the performance on the test peaks.
    Parameters
    ----------
    exp_name
    peaks_df: list[pd.DataFrame]
        List of dataframes of all the genes.
    train_peaks: List[str]
        Name of the genses used for training
    test_peaks: Optional[List[str]]
        Name of the genses used for testing / AL
    data_cfg:DataConfig,
        Config containing all the paratmets of the experiment concerning data
    n_trials:int
        Number of times the experiemnt should be run with differtn hp.
    study_name: str
        Name of the database created for the hpo.
    use_al: bool
        Wheter to use AL during the experiment.
    metric_type: str
        Which metric is tried to be maximized.
        
    Returns
        -------
    float
        Metric of the model computed on the test peaks.
    """
    
    def objective(trial: optuna.Trial):

        """
        Performs a run of training using different hp to find the best configuration
        ----------
        trial: optuna.trial.Trial
            Current active hpo trial
        Returns
            -------
        float
            Metric of the model computed on the test peaks.
        """
        
        
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        epochs = trial.suggest_int("epochs", 5, 15)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.4)
        val_percentage = trial.suggest_float("val_percentage", 0.01, 0.1)
        hidden_channels = trial.suggest_categorical("hidden_channels", [256, 512, 1024, 2048])
        ff_channel = trial.suggest_int("ff_channel", 1,5)
        cnn_channel = trial.suggest_int("cnn_channel",1,5)
        
    
        starting_epoch_multiplier =  trial.suggest_int("starting_epoch_multiplier", 5, 15)
        new_query_weight = trial.suggest_float("new_query_weight", 2.0, 10.0)
        alpha =  trial.suggest_float("alpha", 0.0, 1.0)
    

        train_cfg = TrainConfig(
            lr=lr,
            weight_decay=weight_decay,
            epochs=epochs,
            loss_weight=1,
            use_al = use_al,
            starting_epoch_multiplier=starting_epoch_multiplier,
            new_query_weight=new_query_weight
        )

        model_cfg = ModelConfig(
            dropout_rate=dropout_rate,
            hidden_channels=hidden_channels,
            depth_conv = cnn_channel,
            depth_ffnn = ff_channel,
        )

        al_cfg = ActiveLearningConfig(alpha=alpha, val_percentage=val_percentage)
    
    
        cfg = Config(data_cfg=data_cfg,train_cfg=train_cfg,model_cfg=model_cfg,al_cfg=al_cfg)
    
        if eval_mult:
            metric = evaluate_all_peaks(peaks_df=peaks_df[:-1],cfg=cfg, trial=trial, metric=metric_type)
        else:

            metric = run_training(peaks_df=peaks_df, train_peaks=train_peaks, test_peaks=test_peaks, cfg=cfg, trial=trial)[metric_type]
        
        
        gc.collect()
        torch.cuda.empty_cache()
        
        return metric

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(multivariate=True),
        pruner=HyperbandPruner(),
        study_name=study_name,
        storage=f"sqlite:///{study_name}_{metric_type}.db",
        load_if_exists=True,
    )

    study.optimize(
        objective,
        n_trials=n_trials,
        gc_after_trial=True,  
    )

    best_trial = study.best_trial

    print("\nBest trial:")
    print(f"  Value: {best_trial.value}")
    print("  Params:")
    for key, value in best_trial.params.items():
        print(f"    {key}: {value}")

    return best_trial.params  