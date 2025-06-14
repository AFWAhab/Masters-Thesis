import matplotlib.pyplot as plt
import numpy as np

def get_data_for_file(file_path):
    with open(file_path, 'r') as file:
        total_bases = 0
        n_bases = 0
        for line in file:
            if ">" in line:
                continue
            stripped_line = line.rstrip()
            stripped_line = stripped_line.replace("\n", "")
            total_bases += len(stripped_line)
            n_bases += stripped_line.upper().count('N')
        return float(n_bases) / float(total_bases)

if __name__ == '__main__':
    base_file_name = "../data/pig/raw_chromosomes/Sus_scrofa.Sscrofa11.1.dna.primary_assembly."
    all_file_names = []
    file_names_short = []
    data = []
    for i in range(18):
        all_file_names.append(f"{base_file_name}{i + 1}.fa")
        file_names_short.append(f"{i + 1}")
        data.append(get_data_for_file(all_file_names[i]))
    all_file_names.append(f"{base_file_name}X.fa")
    file_names_short.append("X")
    data.append(get_data_for_file(all_file_names[18]))
    y_pos = np.arange(len(file_names_short))
    _, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlabel("Chromosome")
    ax.set_ylabel("N bases / total bases ratio")
    ax.set_title("Ratio of unknown to known bases in raw chromosome data")

    plt.bar(y_pos, data)
    plt.xticks(y_pos, file_names_short)


    plt.savefig("plots/N_ratio.png")


