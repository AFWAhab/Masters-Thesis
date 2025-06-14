import math
import os.path
import random
import pandas as pd

import numpy as np
from Bio import SeqIO
import h5py

file_prefix = "../data/pig/fasta_output/10k-corrected/"
out_prefix = "../data/pig/out/10k-corrected-half-life/matched/display/"
should_match = False
PROCESS_FLAGS = False

# file_prefix = "../data/zebrafish/fasta_output/"
# out_prefix = "../data/zebrafish/out/"

number_base_pairs = 7000 + 3500


def parse_fasta(filename):
    fasta_sequences = SeqIO.parse(open(filename), 'fasta')
    res = {}
    for fasta in fasta_sequences:
        name, sequence = str(fasta.id), str(fasta.seq)
        namesplit = name.split("|")[0]
        namesplit = namesplit.replace('>', '')
        res[namesplit] = sequence
    return res


def extract_sequences():
    return parse_fasta(file_prefix + "combined.fa")


def extract_expression_values():
    res = {}
    with open("../data/pig/pig_expression.txt", 'r') as file:
        # with open("../data/zebrafish/zebrafish_promoter_sequences_expression.txt", 'r') as file:
        for line in file:
            if "Gene ID,Median log RPKM" in line:
                continue
            split = line.split(',')
            res[split[0]] = float(split[1])
    return res


def create_one_hot_encoding(acid):
    if acid == 'A':
        return [True, False, False, False]
    elif acid == "C":
        return [False, True, False, False]
    elif acid == 'G':
        return [False, False, True, False]
    elif acid == 'T':
        return [False, False, False, True]
    else:
        return [False, False, False, False]


# Simple but might cause data type problems as we aren't quite the same string format (|0 vs <f8) as the original data
def create_geneName(file, gene_ids):
    file.create_dataset("geneName", data=gene_ids)


def create_label(file, ordered_expression_values):
    file.create_dataset("label", data=ordered_expression_values)


def create_flags(file, ordered_flags):
    file.create_dataset("flags", data=ordered_flags)


def create_and_save_half_life_data(data, cachefile):
    if os.path.isfile(cachefile):
        return
    with open(cachefile, 'w') as file:
        prefix = ""
        for value in data[0]:
            file.write(prefix)
            file.write(str(value.item()))
            prefix = ","


def create_and_save_gene_id_array(fasta_sequence, cachefile):
    if os.path.isfile(cachefile):
        return
    res = []
    i = 0
    for acid in fasta_sequence:
        res.append(create_one_hot_encoding(acid))
        i += 1
    with open(cachefile, "w") as file:
        for arr in res:
            file.write(str(arr))
            file.write("\n")
        file.close()


def read_cached_half_life_data(gene_id):
    res = []
    if not os.path.isfile("cache/display/" + gene_id):
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    print(f"Data found for gene id: {gene_id}")
    with open("cache/10k-corrected-half-life/" + gene_id) as file:
        for line in file:
            if line == "":
                continue
            split = line.split(",")
            for value in split:
                res.append(float(value))
    return res


def read_cached_data(gene_id, max_size):
    res = []
    with open("cache/10k-corrected/" + gene_id) as file:
        for line in file:
            if len(res) == max_size:
                return res
            if line == "":
                continue
            line = line.replace('[', '')
            line = line.replace(']', '')
            line = line.replace('\n', '')
            split = line.split(",")
            encoding = []
            for val in split:
                if "False" in val:
                    encoding.append(False)
                elif "True" in val:
                    encoding.append(True)
                else:
                    raise "Unknown value parsed from cache, expected one of [True, False] received: " + str(val)
            res.append(encoding)
        file.close()
    return res


def save_halflife_dataset(file, ordered_gene_ids):
    length = len(ordered_gene_ids)
    dset = file.create_dataset("data", (length, 8))
    i = 0
    for gene_id in ordered_gene_ids:
        half_life_data = read_cached_half_life_data(gene_id)
        assert len(half_life_data) == 8
        dset[i] = half_life_data
        i += 1


def save_dataset(file, ordered_gene_ids):
    length = len(ordered_gene_ids)
    dset = file.create_dataset("promoter", (length, number_base_pairs, 4))
    i = 0
    for gene_id in ordered_gene_ids:
        encodings = read_cached_data(gene_id, number_base_pairs)
        dset[i] = encodings
        i += 1


def extract_gene_half_life_data(gene_id, df):
    columns_to_extract = [
        "5' UTR len",
        "3' UTR len",
        "ORF len",
        "5' UTR gc",
        "3' UTR gc",
        "ORF gc",
        "Intron len",
        "Exon junction density"
    ]
    filtered = df[df['Gene ID'] == gene_id][columns_to_extract]
    if filtered.empty:
        print(f"DATAFRAME EMPTY FOR GENE ID {gene_id}")
    return filtered.astype(float).to_numpy()


def create_half_life_data(file, ordered_gene_ids):
    df = pd.read_csv("../analysis/pig_halflife_data.csv", sep=',', low_memory=False)
    print("Started half life creation")
    i = 0
    for gene_id in ordered_gene_ids:
        if i % 100 == 0:
            print(f"Iteration: {i}")
        data = extract_gene_half_life_data(gene_id, df)
        if data.size == 0:
            panic("No halflife data found")
        create_and_save_half_life_data(data, "cache/10k-corrected-half-life/" + str(gene_id))
        i += 1

    save_halflife_dataset(file, ordered_gene_ids)


def create_promoter(file, ordered_gene_ids, fasta_dict):
    print("Started created promoter, expecting ", len(ordered_gene_ids), " iterations")
    i = 0
    for gene_id in ordered_gene_ids:
        if i % 100 == 0:
            print("Iterations: ", i)
        create_and_save_gene_id_array(fasta_dict[gene_id], "cache/10k-corrected/" + str(gene_id))
        i += 1

    save_dataset(file, ordered_gene_ids)


def get_gene_id_flag_map():
    result = {}
    df = pd.read_csv("../data/pig/bgee/data/combined/consolidated_samples.tsv", sep='\t', low_memory=False)
    for _, row in df.iterrows():
        gene_id = row["Gene ID"]
        flag = row["Present"]
        result[gene_id] = bool(flag)
    return result


def create_h5_file(filename, sequences, expression_values):
    file = h5py.File(filename, "w")
    gene_ids = []
    ordered_expression_values = []
    should_use_flags = False
    all_flags = None
    if PROCESS_FLAGS:
        try:
            ordered_flags = []
            all_flags = get_gene_id_flag_map()
        except:
            print("could not load flag data")
        if all_flags is not None:
            if len(all_flags) > 0:
                should_use_flags = True

    for gene_id in sequences.keys():
        expression_value = expression_values[gene_id]
        gene_ids.append(gene_id)
        ordered_expression_values.append(expression_value)
        if should_use_flags:
            ordered_flags.append(all_flags[gene_id])

    print("Creating half-life-data")
    create_half_life_data(file, gene_ids)
    print("Finished half-life data")
    print("Creating gene_name")
    create_geneName(file, gene_ids)
    print("Finished gene_name")
    print("Creating label")
    create_label(file, ordered_expression_values)
    print("Finished label")
    if should_use_flags:
        create_flags(file, ordered_flags)
    print("Creating promoter")
    create_promoter(file, gene_ids, sequences)
    print("Finished promoter")


def match_other_embedding():
    directory_to_match = "../data/pig/out/10k-corrected/"
    test_file = h5py.File(directory_to_match + "test.h5")
    train_file = h5py.File(directory_to_match + "train.h5")
    valid_file = h5py.File(directory_to_match + "valid.h5")

    test_ids = []
    train_ids = []
    valid_ids = []
    for test_id in test_file["geneName"]:
        test_ids.append(test_id.decode('utf-8'))

    for train_id in train_file["geneName"]:
        train_ids.append(train_id.decode('utf-8'))

    for valid_id in valid_file["geneName"]:
        valid_ids.append(valid_id.decode('utf-8'))

    print(len(train_ids))
    print(len(test_ids))
    print(len(valid_ids))
    return train_ids, test_ids, valid_ids


def create_h5_files():
    training_prop = 0.8
    test_prop = 0.1
    valid_prop = 0.1
    max_test_size = 10
    max_valid_size = 10
    assert math.isclose(training_prop + test_prop + valid_prop, 1)
    sequences = extract_sequences()

    training = {}
    test = {}
    valid = {}

    if should_match:
        matched_training, matched_test, matched_valid = match_other_embedding()

    expression_values = extract_expression_values()
    if should_match:
        print(f"{len(expression_values.keys())}, {len(matched_training) + len(matched_test) + len(matched_valid)}")
        valid_mismatch = 0
        test_mismatch = 0
        for value in matched_test:
            if value not in sequences.keys():
                test_mismatch += 1
        for value in matched_valid:
            if value not in sequences.keys():
                valid_mismatch += 1
        print(f"MISMATCH VALID: {valid_mismatch}")
        print(f"MISMATCH TEST: {test_mismatch}")

    mismatch = 0

    if not should_match:
        for name, sequence in sequences.items():
            if name not in expression_values.keys():
                print("Found invalid key :(")
            r = random.uniform(0, 1.0)
            if r <= training_prop:
                training[name] = sequence
            elif r <= training_prop + test_prop:
                if len(test) >= max_test_size:
                    training[name] = sequence
                else:
                    test[name] = sequence
            else:
                if len(valid) >= max_valid_size:
                    training[name] = sequence
                else:
                    valid[name] = sequence
    else:
        for gene_id in matched_training:
            training[gene_id] = sequences[gene_id]
        for gene_id in matched_valid:
            valid[gene_id] = sequences[gene_id]
        for gene_id in matched_test:
            test[gene_id] = sequences[gene_id]
    print(len(training.keys()))
    print(len(test.keys()))
    print(len(valid.keys()))
    print(mismatch)

    print("__Data size__")
    print("Training: ", len(training))
    print("Test: ", len(test))
    print("Valid: ", len(valid))

    print("Creating train.h5")
    # create_h5_file(out_prefix + "train.h5", training, expression_values)
    print("Creating test.h5")
    create_h5_file(out_prefix + "test.h5", test, expression_values)
    print("Creating valid.h5")
    create_h5_file(out_prefix + "valid.h5", valid, expression_values)


def h5test():
    file = h5py.File("../root/autodl-tmp/datasets/embeddings/valid.h5", 'r')
    print(list(file.keys()))
    print(file["data"])
    print(file["geneName"])
    print(file["label"])
    print(file["promoter"])


def createdh5Test():
    file = h5py.File("valid.h5", 'r')
    print(list(file.keys()))
    # print(file["data"])
    print(file["geneName"])
    print(file["label"])
    print(file["promoter"])


if __name__ == '__main__':
    h5test()
    create_h5_files()
    # createdh5Test()
