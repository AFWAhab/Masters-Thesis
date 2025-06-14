import pandas as pd
import os
import random

def get_valid_genes():
    # Select all the genes that we have TSS data for
    df = pd.read_csv("../data/pig/pig_tss_CORRECTED.csv")
    result = []
    for geneID in df["Gene ID"]:
        result.append(geneID)
    return result

def create_combined_tsv(valid_genes):
    directory_path = "../data/pig/bgee/data/"
    all_file_names = os.listdir(directory_path)
    res_df = None
    for idx, file_name in enumerate(all_file_names):
        print(f"Considering file {file_name} ({idx}/{len(all_file_names)})")
        if "combined" in file_name:
            continue
        df = pd.read_csv(directory_path + file_name, sep='\t', low_memory=False)
        df = df[df["Gene ID"].isin(valid_genes)]
        if res_df is None:
            res_df = df
        else:
            res_df = pd.concat([res_df, df])


    res_df.to_csv(f"{directory_path}/combined/all_samples.tsv", sep="\t", encoding="utf-8")

def sanity_check():
    directory_path = "../data/pig/bgee/data/"
    all_file_names = os.listdir(directory_path)
    all_valid_genes = get_valid_genes()
    valid_genes = random.sample(all_valid_genes, 20)
    valid_gene_counts = {}
    for idx, file_name in enumerate(all_file_names):
        print(f"Considering file {idx}/{len(all_file_names)}")
        if "combined" in file_name:
            continue
        df = pd.read_csv(directory_path + file_name, sep='\t', low_memory=False)
        for gene in valid_genes:
            if gene not in valid_gene_counts.keys():
                valid_gene_counts[gene] = 0
            count = (df["Gene ID"] == gene).sum()
            valid_gene_counts[gene] += count

    actual_gene_counts = {}
    df = pd.read_csv(directory_path + "/combined/all_samples.tsv", sep='\t', low_memory=False)
    for gene in valid_genes:
        if gene not in actual_gene_counts.keys():
            actual_gene_counts[gene] = 0
        count = (df["Gene ID"] == gene).sum()
        actual_gene_counts[gene] += count

    for (expected_key, expected_value), (actual_key, actual_value) in zip(valid_gene_counts.items(), actual_gene_counts.items()):
        assert expected_key == actual_key
        if expected_value != actual_value:
            print(f"Mismatch for gene {actual_key}, expected: {expected_value} actual: {actual_value}")
        else:
            print(f"Gene {actual_key} is valid ({expected_value}/{actual_value})")

def consolidate():
    df = pd.read_csv("../data/pig/bgee/data/combined/all_samples.tsv", sep='\t', low_memory=False)
    filtered_df = df[["Gene ID", "Detection flag"]]

    ids = []
    detection_flag = []

    data = {}
    for _, row in filtered_df.iterrows():
        gene_id = row["Gene ID"]
        flag = row["Detection flag"]
        if gene_id not in data.keys():
            data[gene_id] = []
        data[gene_id].append(flag)

    for gene_id, flag_list in data.items():
        present_flags = flag_list.count("present")
        absent_flags = flag_list.count("absent")
        ids.append(gene_id)
        if present_flags >= absent_flags:
            detection_flag.append(True)
        else:
            detection_flag.append(False)
    res_df = pd.DataFrame({
        "Gene ID": ids,
        "Present": detection_flag
    })
    res_df.to_csv("../data/pig/bgee/data/combined/consolidated_samples.tsv", sep='\t', encoding="utf-8")


if __name__ == '__main__':
    #create_combined_tsv(get_valid_genes())
    #sanity_check()
    consolidate()