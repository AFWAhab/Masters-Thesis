import os
import pandas
import pandas as pd
import re
import matplotlib.pyplot as plt
import numpy as np


def get_file_name(all_files, base_name, index):
    file_prefix = base_name + str(index)
    for file in all_files:
        if file_prefix in file:
            return file
    return None

def validate_csv(csv_file_name):
    if csv_file_name is None:
        return False
    df = pd.read_csv("run_data/" + csv_file_name)
    if len(df) > 90:
        return True
    return False

def get_next_name(name, existing_names):
    if name not in existing_names:
        return name

    # Check if name has numbered suffix
    regex = r"\((\d+)\)$"
    match = re.search(regex, name)
    if not match:
        return get_next_name(name + " (1)", existing_names)
    # At this point we know that we have (X) at the end of the string - find the number X and increment it
    number = int(match.group(1))
    new_number = number + 1
    new_name = re.sub(regex, f"({new_number})", name)
    return get_next_name(new_name, existing_names)



def get_file_label(existing_labels, file_name):
    file_name = file_name.replace(".csv", "")
    stripped_name = file_name.replace("FLAG_TEST-", "")
    if stripped_name.isdigit():
        return get_next_name("Base", existing_labels)
    stripped_name = re.sub(r"^FLAG_TEST-\d+-?", "", file_name)
    return get_next_name(stripped_name, existing_labels)

def extract_run_data(csv_file):
    df = pd.read_csv("run_data/" + csv_file)
    highest_acc_row = df.loc[df['ACC'].idxmax()]
    return highest_acc_row

def convert_to_float(value):
    if type(value) == str:
        if "ACC:" in value:
            raw_value = value.replace("ACC:", "")
            return float(raw_value)
        if value == '-':
            return 0.0
    return float(value)

# Extract ACC/PREC/REC/TNR/NPV
def extract_data_from_row(row):
    result = [row["ACC"], row["PREC"], row["REC"], row["TNR"], row["NPV"]]
    float_result = [convert_to_float(unknown_value) for unknown_value in result]
    return result

def draw_graph(csv_data_map, csv_label_map):
    groups = {
        #"Group 1": [0.751000, 0.824500, 0.887400, 0.103400, 0.162200],
        #"Group 2": [0.800000, 0.810000, 0.900000, 0.110000, 0.170000],
        #"Group 3": [0.770000, 0.830000, 0.880000, 0.120000, 0.180000],
    }

    # Taken from https://www.learnui.design/tools/data-color-picker.html
    custom_colors = {
        'ACC': '#003f5c',
        'PREC': '#58508d',
        'REC': '#bc5090',
        'TNR': '#ff6361',
        'NPV': '#ffa600',
    }

    index = 0
    to_skip = 2
    skipped = 0
    run_6_skipped = False
    for csv_file in csv_data_map.keys():
        if skipped < to_skip:
            skipped += 1
            continue
        if index == 6 and not run_6_skipped:
            run_6_skipped = True
            continue
        data = csv_data_map[csv_file]
        label = csv_label_map[csv_file]
        print(f"DATA FOR: {csv_file}")
        row_data = extract_data_from_row(data)
        groups[index] = row_data
        print(f"Index ({index}): {label}")
        index += 1


    metrics = ['ACC', 'PREC', 'REC', 'TNR', 'NPV']

    num_groups = len(groups)
    num_metrics = len(metrics)

    group_spacing = 1.5
    indices = np.arange(num_groups) * group_spacing
    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [groups[group][i] for group in groups]
        bar_positions = indices + i * bar_width
        bars = ax.bar(bar_positions, values, width=bar_width, label=metric, color=custom_colors[metric])

        # Add value labels on top of each bar
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # x position (center of bar)
                height + 0.01,  # y position (top of bar)
                f'{height:.3f}',  # label text
                ha='center', va='bottom', fontsize=10,
                rotation=90
            )

    ax.set_xticks(indices + bar_width * (num_metrics - 1) / 2)
    ax.set_xticklabels(groups.keys())

    ax.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Threshold = 0.5')
    ax.set_ylabel('Value')
    ax.set_xlabel("Run Number")
    ax.set_ylim(top=1.1)
    ax.set_title('Confusion Matrix Comparison for Best Accuracy of Selected Runs')
    ax.legend(title='Metrics', bbox_to_anchor=(1.15, 1), loc='upper right')

    plt.tight_layout()
    plt.show()

def analyse_binary_classifier():
    first_csv = 2
    final_csv = 32
    base_name = "FLAG_TEST-"
    EXCLUDED_FILES = ["FLAG_TEST-10.csv"]
    all_file_names = os.listdir("run_data")
    all_file_names = [x for x in all_file_names if x not in EXCLUDED_FILES]
    valid_csv_files = []
    csv_file_label_map = {}
    for i in range(first_csv, final_csv + 1):
        csv_file_name = get_file_name(all_file_names, base_name, i)
        if validate_csv(csv_file_name):
            valid_csv_files.append(csv_file_name)
    for valid_csv in valid_csv_files:
        csv_file_label_map[valid_csv] = get_file_label(list(csv_file_label_map.values()), valid_csv)

    csv_data_map = {}
    for csv_file in csv_file_label_map.keys():
        data = extract_run_data(csv_file)
        csv_data_map[csv_file] = data

    draw_graph(csv_data_map, csv_file_label_map)

if __name__ == '__main__':
    analyse_binary_classifier()