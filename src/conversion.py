from data.processing import ParseAll

if __name__ == '__main__':
    subset_name = "autotagging_top50tags"
    subset_file = f"./mtgjamendodataset/data/{subset_name}.tsv"
    data_directory = "../mtg-jamendo/"
    output_directory = f"D:/SongsDataset/melspec-mtg-jamendo"
    subset_data = f'./mtgjamendodataset/stats/{subset_name}/all.tsv'

    ParseAll(subset_file, subset_data, data_directory, output_directory)