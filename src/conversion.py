from data.processing import ParseBalanced

if __name__ == '__main__':
    subset_name = "autotagging_top50tags"
    subset_file = f"E:/mtg-jamendo-dataset/data/{subset_name}.tsv"
    data_directory = "E:/mtg-jamendo/"
    output_directory = f"D:/SongsDataset/melspec-mtg-jamendo"
    subset_data = f'E:/mtg-jamendo-dataset/stats/{subset_name}/all.tsv'

    ParseBalanced(subset_file, subset_data, data_directory, output_directory)