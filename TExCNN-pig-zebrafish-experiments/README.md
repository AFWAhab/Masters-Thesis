# This directory holds the code used for section 3.5 of the thesis
This directory was originally based on the TExCNN repository downloadble from https://www.healthinformaticslab.org/supp/resources.php (Puplication: https://www.mdpi.com/2073-4425/15/12/1593). We have only reuploaded files from this repository if we have made changes to them.

Since the original repository grew big an convoluted, duplicate files have been condensed and moved to this repository. As a result, not all scripts are expected to work out of the box.
Specifically, any paths referenced in the files will have to be manually changed.

# Overview of the file structure in this directory
For a quick overview of the fiels in this directory see below
```
│   README.md
│   requirements.txt
│
├───analysis
│   │   analyse_run_data.py
│   │   count_genes.py
│   │   count_samples.py
│   │   gtf2df.py
│   │   halflife.py
│   │   verify_balanced_data.py
│   │   verify_N_bases.py
│   │
│   └───out
│           balanced_output.txt
│           Confusion_matrix_graph.png
│           count_samples_output.txt
│           N_ratio.png
│           pig_halflife_data.csv
│
├───datascripts
│   │   extract_expression_flags.py
│   │   fasta_file_N.py
│   │   generate_embeddings.py
│   │   generate_fasta_file.py
│   │   generate_h5_files.py
│   │
│   ├───in
│   │       pig_tss_CORRECTED.csv
│   │
│   └───out
│           combined-example.fa
│           consolidated_samples.tsv
│           test.h5
│
└───model
    │   train_optuna.py
    │   train_optuna_flags.py
    │   train_optuna_half-life.py
    │
    └───out
            FLAG_TEST-32-positive-negative-balanced-threshold-tuning-mean-all.csv
            Parameters.db
            pigs-half-life-new-embeddings-2.csv
```
## Detailed explanation of each file and associated output

### Analysis
`analysis/analyse_run_data.py`: Used to analyse the .csv file output from a binary classifier optimisation. An example output is shown in `analysis/out/Confusion_matrix_graph.png`

`analysis/count_genes.py`: Simple script to count the number of genes in a fasta file

`analysis/count_samples.py`: Counts and displays which samples are used from multiple RNA-Seq files. Example output shown in `analysis/out/count_sample_output.txt`

`analysis/gtf2df.py`: Helper function copied from https://gist.github.com/rf-santos/22f521c62ca2f85ac9582bf0d91e4054

`analysis/halflife.py`: Script for extracting half-life features from a .gtf file. Example output is shown in `out/pig_halflife_data.csv`

`analysis/verify_balanced_data.py`: Simple script for checking the number of present/absent values from a sanitised collection of flags. Input to this script is the output of the script `datascripts/extract_expression_fags.py`. An example output can be seen in `analysis/out/balanced_output.txt`

`analysis/verify_N_bases.py`: Given a directory of assembly fasta files, this script reads through each files, counting the number of time `N` appear in all of these files, it then graphs the ratio of `N` bases to total bases in each chromosome. An example output can be seen in `analysis/out/N_ratio.png` 

### Datascripts

`datascripts/extract_expression_fags.py`: Using a directory of all RNA-Seq Bgee data (For example: https://www.bgee.org/species/9823), it extracts all samples and does a majority vote on the presence or absence of a gene. It then creates a consolidated representation that can later be used by for example `analysis/verify_balanced_data.py`. An example output can be seen in `datascripts/out/consolidated_samples.tsv`. Before consolidating, the script first combines all sample files into a single `all_samples.tsv` file. However, both the input files and this intermediary file are too large to upload.

`datascripts/generate_fasta_file.py`: Reads TSS information from a file with extracted chromsome/TSS start position/stand data (Example of input data shown in `datascripts/in/pig_tss_CORRECTED.csv`), as well as a directory containing raw fasta files for each chromosome of the genome. Using this data, the script extracts a promoter sequence of the desired length (both upstream/downstream length can be customized), and correctly adjusts for positive/negative strand TSS. The script then outputs a single combined fasta file containing only the promoter sequence for each gene. The variable LAST_GENE_ID can be set to manually skip to a specific gene if the process was stopped during execution. An example of the output (showing only 2 genes) can be found in `datascripts/out/combined-example.fa`

`datascripts/fasta_file_N.py`: Identical to `datascripts/generate_fasta_file.py` except it also tracks the number of `N` bases found, we used this script to determine that there were no `N` bases present in the extracted promoter sequences.

`datascripts/generate_h5_files.py`: Generates .h5 files for training and later embedding creation. This creates files `test.h5`, `train.h5`, and `valid.h5`, ensuring test and validation sets are always exactly 1000 entries large, and putting the rest of the values in the training dataset. The script can handle half-life data if it is present. As input the script uses the combined promoter sequence fasta files as shown in `datascripts/out/combined-example.fa`. Although the output files are too large to display here, we have created a `test.h5` file without half-life data containing only 10 sequences. This file is available in `datascripts/out/test.h5`. If flag data should be processed (for the "present"/"absent" case), then set `PROCESS_FLAGS = True`.

`datascripts/generate_embeddings.py`: Using the .h5 files generated from `datascripts/generate_h5_files.py`, this generates embeddings using DNABERT-2 that we will later use. It uses sliding windows of customizable size as described in the thesis. For example, using 18 windows for 10.5kb and 60 windows for 31.5kb. The output embeddings are too large to display in this repository.

### Model

All the scripts in the `model` sub-directory use similar code with minor changes to accommodate different experiments. This means that most of the code is duplicated.

`model/train_optuna.py`: Trains the CNN on the embeddings and .h5 files. This is the main training file all other experiments are based on. It will train for a customizable number of iterations, and optimise given hyper-parameters using Optuna, the output is recorded in two ways: Each full execution of the scripts creates a .csv file of the run (an example of this is shown in `model/out/pigs-half-life-new-embeddings-2.csv`), but is also stored in the database shown in `model/out/Parameters.db`. To view most of our successful and failed runs, go to https://optuna.github.io/optuna-dashboard/ and upload the `Parameters.db` file, this will then show all of the runs done. This model encapsulates both pig and zebrafish experiments, as it is only a matter of changing the input data.

`model/train_optuna-half_life.py`: The extended script used for training on embeddings that include half-life data. This script also includes the additional MLP described in the thesis for training on only the half-life data

`model/train_optuna_flags.py`: The script used for the modified task of predicting "present"/"absent" flags. Because the final layer of the model is changed, we have redone how the .csv output looks, primarily to include confusion matrix statistics. And example of this new output is shown in `model/out/FLAG_TEST-32-positive-negative-balanced-threshold-tuning-mean-all.csv`.


# Complete pipeline

There are two major pipelines of interest here, the main pipeline for predicting median expression values, and one for predicting the simplified "present"/"absent" gene expression values. We will describe both here.

The general pipeline is as follows

1. Download raw RNA-Seq files (for example from https://www.bgee.org/species/9823) as well as a representative .gtf file
2. Download raw genome fasta file (for example from https://www.ensembl.org/Sus_scrofa/Info/Index) for all desired chromosomes
3. Run the script `datascripts/generate_fasta_file.py` to generate the combined fasta file
4. Run the script `datascripts/generate_h5_files.py` to generate the needed `train.h5`, `test.h5`, `valid.h5`
5. Run the script `datascripts/generate_embeddings.py` to generate the needed embeddings
6. Using the output of (4) and (5), run the desired model script to train a CNN model on the data

For the simplified case of predicting "present"/"absent" gene expression, the pipeline still requires steps 1-5, however additionally needs steps
7. Run the script `extract_expression_flags.py` to extract the flag values needed
8. Customize and run the `model/train_optuna_flags.py` model

# Requirements to run
It was not possible to extract requirements using pipreqs, so we instead used `pip freeze`, to capture all the installed packages. These are shown in `requirements.txt`.

For CUDA version information, see:
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2022 NVIDIA Corporation
Built on Tue_May__3_19:00:59_Pacific_Daylight_Time_2022
Cuda compilation tools, release 11.7, V11.7.64
Build cuda_11.7.r11.7/compiler.31294372_0
```
Not all versions of python are compatible with the versions of packages needed to run the code in these sections, we used Python 3.10.0.

This project was run on Windows 10 with specs: AMD Ryzen 7 3700X, NVIDIA GeForce RTX 3060, 64GB DDR4 Ram

There may be other indirect requirements not captured above

