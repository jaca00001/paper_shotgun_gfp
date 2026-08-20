import pandas as pd

from src.utls         import get_peaks_sources, setup, create_config, run_training
from src.data_loading import prepare_data_protein, prepare_data_multiple_proteins


def main():
        setup()

        # Load WT cgreGFP data separately so it can be added to the set of protein peaks.
        wt_df_cgreGFP: pd.DataFrame = prepare_data_protein(
                filepath="data/cgreGFP/raw_data/cgreGFPWT-00.csv",
                target_row_label="brightness",
                rescale="log10"
        )

        # Load protein peak data from the specified folder and append the WT cgreGFP data.
        # Each entry in peaks_df corresponds to one protein dataset.
        peaks_df: list[pd.DataFrame] = prepare_data_multiple_proteins(
                folderpath="data/mpcgreGFP/data_raw/",
                target_row_label="log_brightness",
                concat_peaks=False,
                additional_peaks=[wt_df_cgreGFP]
        )

        # Get the names of all peaks and a mapping from peak names to their indices in peaks_df.
        # Specify which peaks should be used for training and testing.
        # If test_data is empty, the training data will be split into training and test sets.
        peak_names, _ = get_peaks_sources(peaks_df=peaks_df)
        train_data = [peak_names[-1]]
        test_data = []

        # Configure the data split, training settings, model and active learning parameters.
        # See configs.py for all available configuration options.
        cfg = create_config(
                data_cfg={
                        "train_size": 0.01,
                        "val_size": 0.1,
                        "test_size": 0.1,
                        "split": "perc",
                },
                train_cfg={
                        "use_al": False,
                        "num_runs": 1
                },
                al_cfg={
                        "acquisition_budget_total": 96 * 10,
                        "rounds": 10
                },
                model_cfg={},
        )

        # Run the training pipeline and save the results to the "results" directory.
        run_training(
                peaks_df=peaks_df,
                train_peaks=train_data,
                test_peaks=test_data,
                cfg=cfg
        )


if __name__ == "__main__":
        main()
