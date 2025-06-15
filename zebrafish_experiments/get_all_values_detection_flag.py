import pandas as pd

df1 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP000447.tsv", sep="\t")
df2 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_ERP121186.tsv", sep="\t")
df3 = pd.read_csv("data/tsvFiles/Danio_rerio_RNA-Seq_read_counts_TPM_SRP044781.tsv", sep="\t")

frames = [df1, df2, df3]
df = pd.concat(frames)
print("Lenght of dataframe:", len(df))

unique_gene_ids = df["Gene ID"].unique()

df1_unique_library_ids = df1["Library ID"].unique()
df2_unique_library_ids = df2["Library ID"].unique()
df3_unique_library_ids = df3["Library ID"].unique()

f = open("data/all_values_detection_flag.txt", "w")
f.write("Gene ID,Majority detection flag,")

for lib_id in df1_unique_library_ids:
    f.write(lib_id + ",")

for lib_id in df2_unique_library_ids:
    f.write(lib_id + ",")

for lib_id in df3_unique_library_ids:
    f.write(lib_id + ",")

f.write("\n")

counter = 0
for gene_id in unique_gene_ids:
    counter += 1
    if counter % 100 == 0:
        print("counter:", counter)

    rows = df[df["Gene ID"] == gene_id]
    rows_detection_flag = rows["Detection flag"]

    if len(rows_detection_flag) != 35:
        continue

    number_of_present = len([flag for flag in rows_detection_flag if flag == "present"])
    number_of_absent = len([flag for flag in rows_detection_flag if flag == "absent"])

    if number_of_absent > number_of_present:
        majority_flag = "absent"
    else:
        majority_flag = "present"

    f.write(gene_id + "," + str(majority_flag) + ",")

    for val in rows_detection_flag:
        f.write(str(val) + ",")

    f.write("\n")