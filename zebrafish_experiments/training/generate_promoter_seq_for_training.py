from generate_promoter_seq import get_promoter_seq

# Paths
genome_dir = "../data/zebrafishGenome"
csv_file_train = "../data/zebrafish_train_tss.csv"
output_fasta_train = "data/zebrafish_promoter_sequences_training.fasta"

# We include 10kb upstream and downstream as the training model will adjust its own start and end from a 10kb large sequence
#get_promoter_seq(csv_file_train, genome_dir, output_fasta_train, 10000, 10000)

csv_file_val = "../data/zebrafish_validation_tss.csv"
output_fasta_val = "data/zebrafish_promoter_sequences_validation.fasta"

get_promoter_seq(csv_file_val, genome_dir, output_fasta_val, 10000, 10000)
