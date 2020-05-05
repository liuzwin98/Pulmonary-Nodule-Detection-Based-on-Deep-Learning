from procedures.datasetBuilder import *


LUNA = 'F:\\LUNA16_Dataset'
CSV_PATH = 'G:\\LUNA_Dataset\\CSVFILES\\annotations.csv'

if __name__ == '__main__':
    # Init dataset builder for creating a dataset of evidence to inject
    print('Initializing Dataset Builder for Evidence Injection')
    builder = Extractor(is_healthy_dataset=False, src_dir=LUNA, coords_csv_path=CSV_PATH, parallelize=False)

    # Extract training instances
    # Source data location and save location is loaded from config.py
    print('Extracting instances...')
    builder.extract(plot=False)

    print('Done.')