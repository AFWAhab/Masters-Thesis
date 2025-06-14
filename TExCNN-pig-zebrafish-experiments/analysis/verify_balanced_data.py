def count_genes():
    present_count = 0
    absent_count = 0
    with open("../data/pig/bgee/data/combined/consolidated_samples.tsv", 'r') as file:
        for line in file:
            if "True" in line:
                present_count += 1
            if "False" in line:
                absent_count += 1

    print(f"Present: {present_count}")
    print(f"Absent: {absent_count}")
    print(f"Present/absent ratio: {float(present_count) / float(absent_count)}")

if __name__ == '__main__':
    count_genes()