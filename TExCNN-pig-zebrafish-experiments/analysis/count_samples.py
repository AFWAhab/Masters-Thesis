import os
import pandas as pd
import pickle

def get_sample_count():
    path = "../data/pig/bgee/data/"
    cache_path = "cache/data_samples.pkl"

    # Load cache if available
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            result_map = pickle.load(f)
        print("Loaded from cache.")
    else:
        all_files_raw = os.listdir(path)
        path = "../data/pig/bgee/"
        all_files = []
        all_files.append("Sus_scrofa_Droplet-Based_SC_RNA-Seq_read_counts_CPM_ERP127795.tsv")
        all_files.append("Sus_scrofa_Droplet-Based_SC_RNA-Seq_read_counts_CPM_ERP127795.tsv")
        for file in all_files_raw:
            all_files.append("data/" + file)
        result_map = {}
        counter = 0
        for file_name in all_files:
            print(f"{counter}/{len(all_files)}")
            counter += 1
            if "combined" in file_name:
                print("Skipping combined file")
                continue
            df = pd.read_csv(os.path.join(path, file_name), sep='\t', low_memory=False)
            df_filtered = df[["Experiment ID", "Anatomical entity name"]]
            result_map[file_name] = df_filtered.drop_duplicates().apply(tuple, axis=1).tolist()

        # Save to cache
        with open(cache_path, "wb") as f:
            pickle.dump(result_map, f)
        print("Cache created.")

    print(result_map)
    final_map = {}
    for full_list in result_map.values():
        anatomical_region = full_list[0][1]
        if anatomical_region not in final_map.keys():
            final_map[anatomical_region] = 1
        else:
            final_map[anatomical_region] += 1
    print(final_map)
    total_samples = 0
    used_samples = 0
    for k, v in sorted(final_map.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v}")
        total_samples += v
        if v > 1:
            used_samples += v
    print(f"Samples: {used_samples}/{total_samples}")

if __name__ == '__main__':
    get_sample_count()