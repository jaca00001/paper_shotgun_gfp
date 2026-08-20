# paper_shotgun_gfp

## Overview

The protein data must be provided as a **CSV or TSV file** containing at least:

- A column containing the **protein sequences**
- A column containing the corresponding **fitness values**
- A column with either the **number of mutations** for each sequence or the **aa_genotype**

Data can be loaded in two ways:

- **Single file:** Use `prepare_data_protein`
- **Multiple files:** Use `prepare_data_multiple_proteins`, which loads data from a folder

## Configuration

Training parameters are defined in the configuration file `configs.py`.

The config file controls the parameters used throughout the training process. See `configs.py` for a complete list of available parameters.

## Training and Test Data

Multiple training and test datasets can be provided at the same time.

- All **training datasets** are merged into a single dataset and used for training.
- **Test datasets** are predicted independently.
- If the test dataset list is empty, the training dataset is split into **training and validation** sets.
- When a test dataset is provided, the configured test size can be ignored and the full test set is used instead.
