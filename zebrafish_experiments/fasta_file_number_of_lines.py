num = len([1 for line in open("training/zebrafish_promoter_sequences_training.fasta") if line.startswith(">")])
print(num)