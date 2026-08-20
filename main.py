import pandas as pd

from src.ploting      import *
from src.data_loading import prepare_data_protein, prepare_data_multiple_proteins
from src.utls         import get_peaks_sources, setup, create_config, run_training

def main():
        setup()

        wt_df_avGFP: pd.DataFrame = prepare_data_protein(filepath="data/avGFP/raw_data/avGFPWT-00.tsv", target_row_label="brightness")
        wt_df_cgreGFP: pd.DataFrame = prepare_data_protein(filepath="data/cgreGFP/raw_data/cgreGFPWT-00.csv", target_row_label="brightness", rescale="log10")
      
        peaks_df: list[pd.DataFrame] = prepare_data_multiple_proteins(folderpath="data/mpcgreGFP/new_data_raw/", target_row_label="log_brightness",
                                                                      concat_peaks=False, additional_peaks=[wt_df_cgreGFP])       
        
        wt_peak, peaks, peak_names = get_peaks_sources(peaks_df=peaks_df)
        
        cfg = create_config(
                data_cfg  = {"train_size": 0.321, "val_size": 0.321, "test_size": 0.321},
                train_cfg = {"use_al": False, "num_runs": 1},
                al_cfg    = {"acquisition_budget_total": 96*10, "rounds": 10},
                model_cfg = {},
        )
        
        run_training(peaks_df=peaks_df, train_peaks=wt_peak, test_peaks=[], cfg=cfg, split="perc")

if __name__ == "__main__":
        main()
        