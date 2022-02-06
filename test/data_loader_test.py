import sys
import os

# Append parent dir to path
sys.path.append("..")

from pascal.data_loader import DataSet

DATASETS_DIR = os.getenv("PASCAL_DATASETS_DIR")

data_set_paths = [
    ("Breast Cancer/breast-cancer-wisconsin.data", "Breast Cancer/breast-cancer-wisconsin.attrs"),
]

def run_tests():
    for ds_path in data_set_paths:
        ds = DataSet.from_csv_file(os.path.join(DATASETS_DIR, ds_path[0]), os.path.join(DATASETS_DIR, ds_path[1]))
        print(ds.data.head(10))
        print(ds.attrs)

if __name__ == "__main__":
    run_tests()