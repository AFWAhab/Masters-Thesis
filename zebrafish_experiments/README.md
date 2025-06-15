# Zebrafish related experiments

Note: many of the data files used to run scripts is missing due the files being too large for GitHub. All the relevant data used is cited in the thesis with links to download pages.

The below text describes the structure of the directory along with the content and purpose of the subfolders.

The files contained in the folder zebrafish_experiments are primarily related to tasks of data processing.

- The subfolder data: data revevant to the experiments

  - datasetsExtractedData: total mapped reads and length of genes in various RNA-seq experiments
  - removedGenes: data related to genes that were excluded when running the human trained Xpresso model on zebrafish genes (see 3.4.3 Selecting genes for analysis in the thesis)
  - 

- The subfolder hic: related to Graph Attention Network Experiments.

- The subfolder paperData: relating to experiments with running the CNN models from the Xpresso on zebrafish data

  - pM10Kb_1KTest: parameters for model trained on human data and corresponding performance on zebrafish genes
  - pretrained_models: models from the Xpresso paper

- The subfolder training contains code relating to training the CNN regressor and classifier.

  - The subsubfolder halflife regression relates to running various regression models using mRNA half-life-related features. 
  - The subsubfolder naive_bayes relates to running naive Bayes models using gene sequence k-mers counts from human, zebrafish and pig genes. 
  - pca contains experiments relating to Principal Component Analysis of k-mers and mRNA half-life-related data
  - random_projections: contains experiments relating to random projections of k-mers and mRNA half-life-related data
  - svm: relates to running support vector machine models using gene sequence k-mer counts from human and zebrafish genes.
  - zebrafish_training_NEW: parameters and test set results from various CNN models using zebrafish data to construct regressor models
  - zebrafish_training_NEW_binary: parameters and test set results from various CNN models using zebrafish data to construct classifiers
