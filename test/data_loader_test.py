import sys
import os
import pandas

# Append parent dir to path
sys.path.append("..")

from pascal.data_loader import DataSet
from pascal.transformation import equal_width_discretization, standardize

DATASETS_DIR = os.getenv("PASCAL_DATASETS_DIR")

data_set_paths = [
    ("Breast Cancer/breast-cancer-wisconsin.data", "Breast Cancer/breast-cancer-wisconsin.attrs"),
]

def run_tests():
    for ds_path in data_set_paths:
        ds = DataSet.from_csv_file(os.path.join(DATASETS_DIR, ds_path[0]), os.path.join(DATASETS_DIR, ds_path[1]))
        ds.replace_undefined_data_points(excluded_cols=["Class"], verbose=False)
        # print(ds.ordinal_encode_column("Class"))
        # ds.data["Class2"] = ds.ordinal_encode("Class")
        # print(ds.data.head(25))
        # print(ds.one_hot_encode("Class").head(25))
        
        eq = equal_width_discretization(ds.data, "Uniformity of Cell Size", 5)
        # [print(q.shape) for q in eq]
        
        # print(pandas.qcut(ds.data["Uniformity of Cell Size"], 50, duplicates="drop").value_counts().sort_index())
        print(standardize(ds.data).head(20))
        # print(ds.data.head(20))

if __name__ == "__main__":
    run_tests()