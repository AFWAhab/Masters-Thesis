if __name__ == '__main__':
    with open("../data/pig/fasta_output/3k/combined.fa", 'r') as file:
        count = 0
        for line in file:
            if '>' in line:
                count += 1
        print(f"Found {count} genes in fasta file")
