from data.processing import ParseAll

if __name__ == '__main__':
    subset_name = "autotagging_top50tags"
    subset_file = f"E:/mtg-jamendo-dataset/data/{subset_name}.tsv"
    data_directory = "E:/mtg-jamendo/"
    output_directory = f"D:/SongsDataset/melspec-mtg-jamendo"
    subset_data = f'E:/mtg-jamendo-dataset/stats/{subset_name}/all.tsv'
    ParseAll(subset_file, subset_data, data_directory, output_directory)