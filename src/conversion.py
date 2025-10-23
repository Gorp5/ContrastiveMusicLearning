from data.processing import ParseBalanced

if __name__ == '__main__':
    subset_name = "autotagging_top50tags"
    subset_file = f"./mtg-jamendo-dataset/data/{subset_name}.tsv"
    data_directory = "../mtg-jamendo/melspec_"
    output_directory = f"D:/SongsDataset/melspec-mtg-jamendo"
    subset_data = f'./mtg-jamendo-dataset/stats/{subset_name}/all.tsv'

    ParseBalanced(subset_file, subset_data, data_directory, output_directory)