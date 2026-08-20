import os
import math
import gzip
import pickle
import regex   as re
import pandas  as pd
import seaborn as sns
import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches

from sklearn.decomposition import PCA
from tqdm                  import tqdm
from pathlib               import Path
from collections           import defaultdict

from src.data_loading      import *
from src.model             import *

def plot_dist_peak(peak_dfs: Union[pd.DataFrame, list[pd.DataFrame]], mut: list=None, mode: str="kde", cols: int=1, save: bool=True)->None:
    """
        Plot fluorescence distributions for one or more peak DataFrames.

        Parameters
        ----------
        peak_dfs : pd.DataFrame or list[pd.DataFrame]
            One DataFrame or a list of DataFrames.
        mut : list[int] or tuple[int], optional
            Mutation range [min, max].
        mode : {"kde", "violin"}
            Type of plot.
        cols : int
            Number of subplot columns.
        save : bool
            Whether to save the figure.
        Returns
        -------
    """

    if isinstance(peak_dfs, pd.DataFrame):
        peak_dfs = [peak_dfs]

    n = len(peak_dfs)
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    axes = axes.flatten()

    for ax, peak_df in zip(axes, peak_dfs):

        name = peak_df["gene"].iloc[0]
        wt_brightness = peak_df.loc[peak_df["aa_genotype_native"] == "wt", "fitness"].iloc[0]
        
        pattern = re.compile(r"[A-Za-z]\d+[A-Za-z]")
        if "num_mutations" not in peak_df.columns:
            peak_df["num_mutations"] = peak_df["aa_genotype_native"].apply(lambda x: 0 if x.strip().upper() == "wt" else len(pattern.findall(x)))
        
        if mut:
            peak_df = peak_df[(peak_df["num_mutations"] >= mut[0]) & (peak_df["num_mutations"] <= mut[1])]

        if mode == "kde":

            sns.kdeplot(
                data=peak_df,
                x="fitness",
                fill=True,
                color="skyblue",
                alpha=0.5,
                linewidth=1.5,
                ax=ax
            )

            ax.axvline(
                wt_brightness,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="WT"
            )

            ax.set_xlabel("Fluorescence (log10 scale)")
            ax.set_ylabel("Density")

        elif mode == "violin":

            sns.violinplot(
                data=peak_df,
                x="gene",
                y="fitness",
                inner=None,
                color="lightgrey",
                cut=0,
                ax=ax
            )

            sns.stripplot(
                data=peak_df,
                x="gene",
                y="fitness",
                alpha=0.3,
                size=2.4,
                jitter=0.4,
                color="red",
                ax=ax
            )

            ax.axhline(
                wt_brightness,
                color="red",
                linestyle="--",
                linewidth=1.5,
                label="WT"
            )

            ax.set_ylabel("Fluorescence (log10 scale)")

        ax.set_title(name)
        ax.legend(loc="upper left")
        sns.despine(ax=ax)
        
    for ax in axes[n:]:
        fig.delaxes(ax)

    plt.tight_layout()

    if save:
        add = "full" if not mut else f"{mut[0]}-{mut[1]}"
        plt.savefig(f"dist_{mode}_{add}.png", dpi=300)

    plt.show()

def plot_pca_3d(peaks_df: list[pd.DataFrame])->None:
    """
        Plots a 3D PCA plot of the data

        Parameters
        ----------
        peak_dfs : pd.DataFrame or list[pd.DataFrame]
            One DataFrame or a list of DataFrames.
        Returns
        -------
    """
    oh = [onehot_encode_sequences(df["sequence"]) for df in peaks_df]
    labels = [df["gene"] for df in peaks_df ]
    
    X = np.concatenate([arr.reshape(arr.shape[0], -1) for arr in oh])
    y = np.concatenate([np.asarray(lb) for lb in labels])

    pca = PCA(n_components=3)
    X_pca = pca.fit_transform(X)

    unique_labels = np.unique(y)
    cmap = plt.get_cmap("tab20", len(unique_labels))

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title("3D PCA plot of the different peaks")
    
    for i, label in enumerate(unique_labels):
            idx = (y == label)
            color = "red" if label == "cgreGFPWT-00" else cmap(i)
            
            ax.scatter(
                    X_pca[idx, 0],
                    X_pca[idx, 1],
                    X_pca[idx, 2],
                    color=color,
                    label=label,
                    alpha=0.7,
                    s=20,
            )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
    ax.set_zlabel(f"PC3 ({pca.explained_variance_ratio_[2]:.1%})")
    
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", title="Peak")
    
    plt.tight_layout()
    
    for angle in range(0, 360, 45):
            ax.view_init(elev=20, azim=angle)
            plt.savefig(f"pca_3d_{angle:03d}.png", dpi=300)
        
def plot_hamming_distances(dfs: list[pd.DataFrame], seq_col: str="sequence", chunk_size: int=512)->None:
        """
        Compute mean/std Hamming distance between all dataset pairs.

        Parameters
        ----------
        dfs : list[pd.DataFrame]
            List of dataframes containing the protein data
        seq_col : str
            Name of the column containing the sequnces
        chunk_size : int
                Number of sequences processed at once to reduce computational load.
        Returns
        -------
        """
    
        labels = [df["gene"].iloc[0] for df in dfs]

        amino_acids: list[str] = list("ACDEFGHIKLMNPQRSTVWY")
        aa2int = {aa: i for i, aa in enumerate(amino_acids)}

        dfs = sorted(dfs, key=lambda df: int(re.search(r"-(\d+)$", df["gene"].iloc[0]).group(1)))
      
        matrices = []
        for df in dfs:

                seqs = df[seq_col].tolist()

                L = len(seqs[0])

                mat = np.array([[aa2int[a] for a in s] for s in seqs], dtype=np.uint8)

                matrices.append(mat)

        n = len(matrices)

        mean_matrix = np.zeros((n, n))
        std_matrix = np.zeros((n, n))

        sampled = []
  
        for i in tqdm.tqdm(range(n)):

                A = matrices[i]

                for j in range(n):

                        B = matrices[j]

                        total_sum: float = 0.0
                        total_sq: float = 0.0
                        total_n: int = 0

                        samples = []

                        for a0 in range(0, len(A), chunk_size):

                                A_chunk = A[a0:a0 + chunk_size]

                                for b0 in range(0, len(B), chunk_size):

                                        B_chunk = B[b0:b0 + chunk_size]

                                        d = (A_chunk[:, None, :] != B_chunk[None, :, :]).sum(axis=2)

                                        flat = d.ravel()

                                        total_sum += flat.sum()
                                        total_sq += np.square(flat).sum()
                                        total_n += flat.size

                                        samples.extend(flat)

                        mean = total_sum / total_n
                        var = total_sq / total_n - mean**2
                        std = np.sqrt(max(var, 0))

                        mean_matrix[i, j] = mean
                        mean_matrix[j, i] = mean

                        std_matrix[i, j] = std
                        std_matrix[j, i] = std
   

                        sampled.append(
                            pd.DataFrame({
                                    "Comparison":
                                    f"{labels[i]} vs {labels[j]}",
                                    "Distance": samples
                            })
                        )

                        print(f"Finished {labels[i]} vs {labels[j]}")

        mean_df = pd.DataFrame(mean_matrix, index=labels, columns=labels)

        std_df = pd.DataFrame(std_matrix, index=labels, columns=labels)

        fig, ax = plt.subplots(1, 2, figsize=(14, 6))

        sns.heatmap(
            mean_df,
            cmap="viridis",
            annot=True,
            fmt=".1f",
            ax=ax[0],
        )
        ax[0].set_title("Mean Hamming distance")

        sns.heatmap(
            std_df,
            cmap="magma",
            annot=True,
            fmt=".1f",
            ax=ax[1],
        )
        ax[1].set_title("Std Hamming distance")

        plt.tight_layout()
        plt.savefig("heatmap_hamming_distances.png", dpi=300)

       

        violin_df = pd.concat(sampled)
        comparisons = violin_df["Comparison"].unique()

        n = len(labels)
        fig, axes = plt.subplots(
                n,
                n,
                figsize=(2*n, 2*n),
                sharex=True,
                sharey=True,
        )

        for comp in comparisons:

                ds1, ds2 = comp.split(" vs ")

                i = labels.index(ds1)
                j = labels.index(ds2)
                ax = axes[i, j]

                sns.violinplot(
                    data=violin_df[violin_df["Comparison"] == comp],
                    y="Distance",
                    inner="quartile",
                    cut=0,
                    linewidth=0.8,
                    ax=ax,
                )

                if i == 0:
                        ax.set_title(ds2, fontsize=20)

                if j == 0:
                        ax.set_ylabel(ds1, fontsize=20)


        plt.tight_layout()
        plt.savefig("violin_grid.png", dpi=200)
                                        
def save_metrics(model: ProtCNN, unlabeled_dataset: torch.utils.data.Dataset, location: str,
                 y_mean: float = 0.0, y_std: float = 1.0, id_plot: Optional[int] = None) -> dict:
    """
    Compute metrics and plots for each unique "gene" in the unlabeled dataset,
    saving them in separate subfolders.

    Parameters
    ----------
    model: ProtCNN
        Model which is evaluated
    unlabeled_dataset: torch.utils.data.Dataset
        Test data on which is model is evaluated on
    location: str
        Save location of the results
    y_mean: float
        Mean of the train set.
    y_std: float
        Std of the train set.
    id_plot: Optional[int]
        Can be used to specify i.e. the round of the active learning process.
    Returns
    -------
    source_metrics : dict
        Dictionary mapping source to the metrics
    """
    
    suffix = f"round_{id_plot}" if id_plot is not None else ""
    base_exp_dir = Path("project/results") / location / "plots" / suffix
    base_exp_dir.mkdir(parents=True, exist_ok=True)

    X_list, y_list, o_list, s_list, epi_list = [], [], [], [], []
    for X, y, o, s, e in unlabeled_dataset:
        X_list.append(X.unsqueeze(0))
        y_list.append(y.unsqueeze(0))
        o_list.append(o.unsqueeze(0))
        s_list.append(s.unsqueeze(0))
        epi_list.append(e.unsqueeze(0))

    X_all = torch.cat(X_list)
    y_all = torch.cat(y_list)
    o_all = torch.cat(o_list)
    s_all = torch.cat(s_list)
    epi_all = torch.stack(epi_list)
    
    unique_sources = torch.unique(s_all).tolist()
    source_metrics = {}
    
    model_path = os.path.join(base_exp_dir, "model.pth")
    dataset_path = os.path.join(base_exp_dir, "unlabeled_dataset.pth")
    mean_std_peath = os.path.join(base_exp_dir, "mean_std.pth")

    for source in unique_sources:
        
        exp_dir = base_exp_dir / f"source_{source}"
        exp_dir.mkdir(parents=True, exist_ok=True)

        idx = (s_all == source).nonzero(as_tuple=True)[0]
        X_source = X_all[idx]
        y_source = y_all[idx]
        o_source = o_all[idx]
        s_source = s_all[idx]
        epi_source = epi_all[idx]

        source_dataset = torch.utils.data.TensorDataset(X_source, y_source, o_source, s_source, epi_source)

        res = evaluate(model, source_dataset, y_mean=y_mean, y_std=y_std)
        source_metrics[source] = res

        log_file = exp_dir / "metrics.txt"
        with log_file.open("w", encoding="utf-8") as f:
            for key, value in res.items():
                f.write(f"{key}: {value}\n")
        
        preds, _, = predict(model, source_dataset, y_mean=y_mean, y_std=y_std)
        pred_unlabeled = np.array(preds).squeeze()

        torch.save(model.state_dict(), model_path)
        torch.save(unlabeled_dataset, dataset_path)
        torch.save([y_mean,y_std], mean_std_peath)
            
        y_true = np.stack([y.cpu().numpy() for _, y, _, _ , _ in source_dataset]).squeeze()
        epi = np.array([e.cpu().numpy() for _, _, _, _ , e in source_dataset]).squeeze()
        plt.figure(figsize=(6, 4))
        sc = plt.scatter(y_true, pred_unlabeled,c = epi, alpha=0.6)
        cbar = plt.colorbar(sc)
        cbar.set_label("Epistatic")
        plt.xlabel("True Fluorescence")
        plt.ylabel("Predicted Fluorescence")
        plt.title(f"Predicted vs True - Source {source}")
        plt.grid(True)
        plt.savefig(exp_dir / f"pred_vs_true.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        sns.kdeplot(y_true, label="True", fill=True)
        sns.kdeplot(pred_unlabeled, label="Predicted", fill=True)
        plt.xlabel("Fluorescence (log)")
        plt.title(f"Distribution Comparison - Source {source}")
        plt.legend()
        plt.savefig(exp_dir / f"brightness_distribution_kde.png")
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.hist(pred_unlabeled, bins=30, alpha=0.7, label="Predicted")
        plt.hist(y_true, bins=30, alpha=0.5, label="True")
        plt.xlabel("Fluorescence (log)")
        plt.ylabel("Frequency")
        plt.title(f"Predicted vs True Brightness - Source {source}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(exp_dir / f"brightness_distribution_unlabeled.png")
        plt.close()

    return source_metrics[source]

def load_results(folder_path: Path)->list:
    """
        Loads the resuslts of a experiment given the path.

        Parameters
        ----------
        folder_path : Path
            Path to the saved experiment
        Returns
        -------
        list
            Metrics of the experiment
    """
    
    
    sub_folders = defaultdict(list)
    
    for experiment in folder_path.iterdir():
            match = re.search(r"(cgre[A-Za-z0-9]+-\d+)", experiment.name)
            if match:
                    sub_folders[match.group(1)].append(experiment)
    
    rows = []
    
    for exp_id, paths in sub_folders.items():
            
            for p in paths:
                    plots_dir = p / "plots"
    
                    if not plots_dir.exists():
                            continue
    
                    for round_dir in  [plots_dir / "round_0", plots_dir / "round_5", plots_dir / "round_10"]: 
                            if not round_dir.is_dir():
                                    continue
                        
                            round_match = re.search(r"round_(\d+)", round_dir.name)
                            round_num = int(round_match.group(1)) if round_match else -1
                            
                            for source_dir in round_dir.iterdir():
                                    metrics_file = source_dir / "metrics.txt"
    
                                    if not metrics_file.exists():
                                            continue
    
                                    metrics = {"exp_id": exp_id}
                                    name = p.name
                                    match = re.search(r"train_([^_]+)", name)
                                    method = match.group(1) if match else "unknown"
    
                                    with open(metrics_file) as f:
                                            for line in f:
                                                    key, value = line.strip().split(":", 1)
                                                    metrics[key.strip()] = float(value.strip())
                                    
                                    metrics["method"] = method
                                    metrics["round"] = round_num
                                    rows.append(metrics)
    
    return rows
    
def plot_dot_plot_compariston(results: list):
    """
       Plots the performance of the two data selecion methods, given the data produced by the load_results function

        Parameters
        ----------
        results : list
            Metrics of the experiment produced by load_results
        Returns
        -------
    """
    
    df = pd.DataFrame(results)

    df = df.sort_values(["exp_id", "method", "round"])

    metrics = ["R2", "Spearman", "Pearson"]
  
    fig, axes = plt.subplots(
        1,
        len(metrics),
        figsize=(10 * len(metrics), 8),
        dpi=200,
    )

    fig.suptitle("Performance Comparison: Standard Mutation vs Shotgun", fontsize=16)

    if len(metrics) == 1:
        axes = [axes]

    shotgun = 0
    

    exp_ids = sorted(df["exp_id"].unique(),key=lambda x: int(x.split("-")[-1]))
    cmap = plt.cm.get_cmap("tab20", len(exp_ids))
    exp_colors = {exp_id: cmap(i)for i, exp_id in enumerate(exp_ids)}

    for ax, metric in zip(axes, metrics):

            pivot = df.pivot_table(
                    index=["exp_id", "round"],
                    columns="method",
                    values=metric,
            )

            pivot = pivot.dropna(subset=["s", "cgreGFPWT-00"])


            if metric == "R2":
                    shotgun = pivot["s"]

       
            rounds = pivot.index.get_level_values("round")

            min_round = rounds.min()
            max_round = rounds.max()

            with gzip.open("hamming_results.pkl.gz", "rb") as f:
                    results = pickle.load(f)

            mean_df = results["mean"]
           
            mean_per_dataset = {
                    idx: mean_df.loc[idx].drop(idx).mean()
                    for idx in mean_df.index
            }
            
        
            for exp_id, group in pivot.groupby(level="exp_id"):
                    
                    group = group.sort_index(level="round")
                    color = exp_colors[exp_id]
                 
                    mean = mean_per_dataset[exp_id]
                    
                    group_rounds = group.index.get_level_values("round")
                    if min_round == max_round:
                            group_sizes = np.full(len(group), 100)
                    else:
                            group_sizes = 40 + (group_rounds/ max_round) * 140

                    ax.plot(
                            group["cgreGFPWT-00"],
                            group["s"],
                            color=color,
                            alpha=0.5,
                            linewidth=1.5,
                            zorder=1,
                    )

                    ax.scatter(
                            group["cgreGFPWT-00"],
                            group["s"],
                            s=group_sizes,
                            color=color,
                            edgecolor="black",
                            linewidth=0.8,
                            alpha=0.85,
                            label=exp_id,
                            zorder=2,
                    )

                    exceptions_R2 = ["cgre3224-18"]
                    exceptions_Spearman = ["cgre1414-48", "cgre9708-12","cgre132-06","cgre2880-14"]
                    exceptions_Pearson = ["cgre9708-12", "cgre575-24","cgre985-36"]
                    move_pearson_x = [ "cgre575-24"]
                    move_pearson_y = ["cgre83-30"]
                    
                    for (_, round_num), row in group.iterrows():

                            if metric == "R2":
                                    exceptions = exceptions_R2
                                    move_y = []

                                    move_x = []
                            elif metric == "Spearman":
                                    exceptions = exceptions_Spearman
                                    move_x = []
                                    move_y = []

                            elif metric == "Pearson":
                                    exceptions = exceptions_Pearson
                                    move_x = move_pearson_x
                                    move_y = move_pearson_y
                            else:
                                    exceptions = []
                                    move_x = []
                                    move_y = []

                            dist_x = 24
                            dist_y = 4
                            x_new =  dist_x if exp_id in move_x else 0
                            y_new =  dist_y if exp_id in move_y else 0
                            xytext = (0, 8) if exp_id in exceptions else (0, -8)
                            xytext = (xytext[0] + x_new, xytext[1] + y_new)
                            
                            ax.annotate(
                                    f"{exp_id}/{int(np.round(mean, 0))}",
                                    (row["cgreGFPWT-00"], row["s"]),
                                    xytext=xytext,
                                    textcoords="offset points",
                                    ha="center",
                                    va="bottom" if exp_id in exceptions else "top",
                                    fontsize=7,
                                    color=color,
                            )

            all_values = pd.concat([pivot["cgreGFPWT-00"], pivot["s"]])

            vmax = 1
            if all_values.min() >= 0:
                    vmin = 0
            else:
                    buffer = 0.02 * (vmax - all_values.min()) 
                    vmin = all_values.min() - buffer

            ax.plot(
                    [vmin, vmax],
                    [vmin, vmax],
                    "k--",
                    alpha=0.2,
                    zorder=1
            )

            ax.set_xlim(vmin, vmax)
            ax.set_ylim(vmin, vmax)
            ax.set_xlabel("Standard Mutation (cgreGFPWT-00)")
            ax.set_ylabel("Shotgun Peaks")
            ax.set_title(metric)

            ax.grid(True, alpha=0.3)


    plt.tight_layout()

    plt.savefig("Performance_Comparison_FT.png", dpi=300, bbox_inches="tight")
    plt.close()

    return pivot["winner"].to_dict(), shotgun.to_dict()

def load_saved_results_config(base_path: Path, cfg: Config=None, al: bool=False)->Tuple[np.ndarray, torch.Tensor]:
    """
        Given a saved model and test datasets, recomputes the predictions using the model

        Parameters
        ----------
        base_path : Path
            Path to the saved experiment
        cfg: Config=None
            Saved config used to load the model
        al: bool
            Used to load the model with the correct hyperparameters.
        Returns
        -------
        list
            Metrics of the experiment
    """
  
    unlab_data = torch.load(base_path /  "unlabeled_dataset.pth", map_location="cuda",  weights_only=False)
    
    if not al:        
        model = ProtCNN(
                        feat_dim=unlab_data[0][0].shape[0],
                        hidden_channels=  512,
                        depth_conv=  4,
                        depth_ffnn=  4,
                        dropout_rate=  0.13249619898782194 
        )
    else:
        model = ProtCNN(
                        feat_dim=unlab_data[0][0].shape[0],
                        hidden_channels=  cfg.model_cfg.hidden_channels,
                        depth_conv=  cfg.model_cfg.depth_conv,
                        depth_ffnn=  cfg.model_cfg.depth_ffnn,
                        dropout_rate=  0.37188 
        )
            

    
    model.load_state_dict(torch.load(base_path /  "model.pth", map_location="cuda", weights_only=False))

    mean, std = torch.load(base_path / "mean_std.pth", map_location="cuda", weights_only=False)

    mean, var =  predict(model = model,test_dataset=unlab_data, y_mean=mean, y_std=std)
    return mean, unlab_data

def plot_multiple_violin(path_one: Path, path_two: Path, cfg: Config=None)->None:
    """
        Plots a violin plot for two different experiments and compares them in the same figure.

        Parameters
        ----------
        path_one : Path
            Path to first experiment
        path_two: Path 
            Path to second experiment
        cfg: Config
            Config containing the hyperparameters
        Returns
        -------
    """
    
    guessed_dist, true_dist = load_saved_results_config(path_one, cfg=cfg, al=False)  
    guessed_dist_2, _  = load_saved_results_config(path_two, cfg=cfg, al=False)      
    guessed_dist = guessed_dist.ravel()
    guessed_dist_2 = guessed_dist_2.ravel()

    peak = re.search(r"(cgre\d+[A-Za-z]?-\d+)", str(path_one)).group(1)
  
    fig, ax = plt.subplots(figsize=(8, 10))

    ymin = min(min(true_dist), min(guessed_dist), min(guessed_dist_2))
    ymax = max(max(true_dist), max(guessed_dist), max(guessed_dist_2))
    
    
    ax.set_ylim(ymin, ymax)
    ax.axvline(0, color="black", lw=2)

    
    plt.title("True vs Predicted Distribution", fontsize=16)
    plt.ylabel("Fluorescence (Log)")
    plt.xlabel("Test Peak")

    sns.violinplot(
        x=np.zeros(len(true_dist)),
        y=true_dist,
        color="lightgray",
        inner=None,
        linewidth=2,
        ax=ax,
        cut=0,
        density_norm="width",
    )

    
    sns.violinplot(
        x=np.zeros(len(guessed_dist)),
        y=guessed_dist,
        color="tab:blue",
        inner=None,
        linewidth=1,
        ax=ax,
        alpha=0.5,
        cut=0,
        density_norm="width",
    )

    sns.violinplot(
        x=np.zeros(len(guessed_dist_2)),
        y=guessed_dist_2,
        color="tab:red",
        inner=None,
        linewidth=1,
        ax=ax,
        alpha=0.5,
        cut=0,
        density_norm="width",
    )

    center = 0
    v = ax.collections[1]
    verts = v.get_paths()[0].vertices
    verts[:, 0] = np.minimum(verts[:, 0], center)

    v = ax.collections[2]
    verts = v.get_paths()[0].vertices
    verts[:, 0] = np.maximum(verts[:, 0], center)
    
    v = ax.collections[1]  
    verts = v.get_paths()[0].vertices

    ax.set_xticks([0])
    ax.set_xticklabels([peak])

    label_one = "Shotgun" if path_one.name.startswith("FT_Mul") else "cgreGFP"
    label_two = "cgreGFP" if label_one  == "Shotgun" else "Shotgun"
    
    true_patch = mpatches.Patch(color="lightgray", label="True distribution")
    guess_patch = mpatches.Patch(color="tab:blue", label=label_one)
    guess2_patch = mpatches.Patch(color="tab:red", label=label_two)
    ax.legend(handles=[true_patch, guess_patch, guess2_patch])

    plt.savefig(f"dist{peak}.png", dpi=300)
    plt.tight_layout()
    plt.show()  

def plot_multiple_kde(path_one: Path, path_two: Path, path_three: Path, path_four: Path, hist: bool=False, cfg: Config=None)->None:
    """
        Plots a kde or histogram plot for two different experiments with and without active learning in the same plot

        Parameters
        ----------
        path_one : Path
            Path to first experiment
        path_two: Path 
            Path to second experiment
        path_three : Path
            Path to first experiment with al
        path_four: Path 
            Path to second experiment with al
        hist: bool
            Whether to use a hist or kde.
        cfg: Config
            Config containing the hyperparameters
        Returns
        -------
    """
    
    
    guessed_dist, data_set = load_saved_results_config(path_one, cfg=cfg, al=False) 
    guessed_dist_2, _  = load_saved_results_config(path_two, cfg=cfg, al=False)       
    guessed_dist = guessed_dist.ravel()
    guessed_dist_2 = guessed_dist_2.ravel()
    
    guessed_dist_3, _  = load_saved_results_config(path_three, cfg=cfg, al=True)       
    guessed_dist_4, _  = load_saved_results_config(path_four, cfg=cfg, al=True)       
    guessed_dist_3 = guessed_dist_3.ravel()
    guessed_dist_4 = guessed_dist_4.ravel()
    
    
    peak = re.search(r"(cgre\d+[A-Za-z]?-\d+)", str(path_one)).group(1)
    
    true_dist = np.stack([y.cpu().numpy() for _, y, _, _ , _ in data_set]).squeeze()
    
    fig, ax = plt.subplots(figsize=(8, 6))

    if hist:
            bins = np.linspace(
                    min(np.concatenate([true_dist, guessed_dist, guessed_dist_2, guessed_dist_3, guessed_dist_4])),
                    max(np.concatenate([true_dist, guessed_dist, guessed_dist_2, guessed_dist_3, guessed_dist_4 ])),
            50,
            )
            
            ax.hist(
                    true_dist,
                    bins=bins,
                    density=True,
                    linewidth=2.5,
                    color="black",
                    label="True distribution",
                    fill=False,
            )

            ax.hist(
                    guessed_dist,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label="cgreGFP",
            )

            ax.hist(
                    guessed_dist_2,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label="Shotgun Peaks",
            )
            
            ax.hist(
                    guessed_dist_3,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label="cgreGFP AL",
            )
            
            ax.hist(
                    guessed_dist_4,
                    bins=bins,
                    density=True,
                    alpha=0.5,
                    label="Shotgun Peaks AL",
            )

    else:
            sns.kdeplot(
                    true_dist,
                    fill=True,
                    color="lightgray",
                    alpha=0.9,
                    linewidth=3,
                    label="True distribution",
                    ax=ax,
            )

            sns.kdeplot(
                    guessed_dist,
                    fill=False,
                    alpha=0.4,
                    linewidth=2,
                    label="cgreGFP",
                    ax=ax,
            )

            sns.kdeplot(
                    guessed_dist_2,
                    fill=False,
                    alpha=0.4,
                    linewidth=2,
                    label="Shotgun Peaks",
                    ax=ax,
            )
            
            sns.kdeplot(
                    guessed_dist_3,
                    fill=False,
                    alpha=0.4,
                    linewidth=2,
                    label="cgreGFP AL",
                    ax=ax,
            )
            
            sns.kdeplot(
                    guessed_dist_4, 
                    fill=False,
                    alpha=0.4,
                    linewidth=2,
                    label="Shotgun Peaks AL",
                    ax=ax,
            )
            
    ax.set_title("True vs Predicted Distributions")
    ax.set_xlabel("Fluorescence (Log)")
    ax.set_ylabel("Density")
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"kde_plot_{peak}.png", dpi=300)

def plot_results_comparison(base_path: Path, al_path: Path, peaks_df: list[pd.DataFrame], hist: bool=False, cfg: Config=None, mode: str="kde")->None:
    """
        Given a path containing multiple experiments each done using both methods, as well as a path containing the same experiments conducted with acive learning
        either plots a violin or kde plot for each.

        Parameters
        ----------
        base_path : Path
            Path to experiments without al
        al_path: Path 
            Path to experiments with al
        peaks_df: list[pd.DataFrame]
            List of df containing the data
        hist: bool
            Whether to use a hist or kde.
        cfg: Config
            Config containing the hyperparameters
        mode: str
            Whether to use kde or violin.
        Returns
        -------
    """

    for p in peaks_df:
            gene = p["gene"].iloc[0]
            
            matches = [
                experiment
                for experiment in base_path.iterdir()
                        if re.search(gene, experiment.name)
            ]
            
            matches_al = [
                    experiment
                    for experiment in al_path.iterdir()
                    if re.search(gene, experiment.name)
            ]
            
            if len(matches) < 2:
                    print(f"Could not find two experiments for {gene}")
                    continue

            if mode == "kde":
                
                if len(matches_al) < 2:
                    print(f"Could not find two experiments for {gene} in AL path")
                    continue
                
                plot_multiple_kde(
                    path_one=matches[0] / "plots" / "round_0",
                    path_two=matches[1] / "plots" / "round_0",
                    path_three =  matches_al[0] / "plots" / "round_10",
                    path_four =   matches_al[1] / "plots" / "round_10",
                    hist = hist,
                    cfg = cfg
                ) 
                
            elif mode == "violin":
                plot_multiple_violin(
                    path_one=matches[0] / "plots" / "round_0",
                    path_two=matches[1] / "plots" / "round_0",
                    cfg = cfg)