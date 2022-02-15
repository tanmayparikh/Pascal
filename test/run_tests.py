import sys
import os

# Append parent dir to path
sys.path.append("..")

import pascal
from pascal.data_loader import DataSet

DATASETS_DIR = os.getenv("PASCAL_DATASETS_DIR")
RANDOM_SEED = 454352534

def pretty_print_bined_data(binned_data, bins):
    for i, bd in enumerate(binned_data):
        num_samples = bd.shape[0]
        if i == len(bins) - 1:
            print(f"Bin{i+1}: {num_samples} Samples\tx ≥ {bins[i]:.4f}")
        else:
            print(f"Bin{i+1}: {num_samples} Samples\t{bins[i]:.4f} ≤ x < {bins[i+1]:.4f}")

def run_abalone_tests(data_file):
    ds = DataSet.from_csv_file(data_file)
    
    print(ds.data.head(5))
    
    print()
    print(ds.summary())
    
    print()
    print("One-hot-encoding `Sex` Variable")
    print(ds.one_hot_encode("Sex").sample(5, random_state=RANDOM_SEED))
    
    print()
    print("Equal width binning on the `Length` variable")
    binned_data, bins = pascal.equal_width_discretization(ds, "Length", 5)
    pretty_print_bined_data(binned_data, bins)
    
    print()
    print("Standardizing all variables except `Sex` and `Rings`")
    cols = ["Length", "Diameter", "Height", "Whole weight", "Shucked weight", "Viscera weight", "Shell weight"]
    data_std = pascal.standardize(ds.data, cols=cols)
    print(data_std.head(5))
    
    print()
    print("Means post standardization")
    print(data_std[cols].mean())
    
    print()
    print("Std Dev post standardization")
    print(data_std[cols].std())
    
    print()
    print("Spitting data into 5-Folds")
    k_fold = pascal.KFoldCrossValidation(ds)
    for i, (train, test) in enumerate(k_fold.split(5)):
        print(f"Fold{i+1}: Test data indices {test[0].head(1).index.item()}-{test[0].tail(1).index.item()}")
    
    print()
    print("Using the AverageRegressor to predict `Rings`")
    avg_reg = pascal.AverageValueRegressor()
    avg_reg.fit(ds)
    pred = avg_reg.predict(ds)
    new_data = ds.data.copy()
    new_data["Prediction"] = pred   
    print(new_data[["Rings", "Prediction"]].sample(10, random_state=RANDOM_SEED)) 
    
    print()
    print("Calculating the Mean Abs Error")
    print(pascal.mean_abs_error(new_data["Rings"], new_data["Prediction"]))
    
def run_breast_cancer_test(data_file):
    ds = DataSet.from_csv_file(data_file)
    print(ds.summary())
    
    print()
    print("Removing undefined data points to fix the `Bare Nuclei` variable being NA")
    ds.replace_undefined_data_points()
    print(ds.summary())

def run_car_evaluation_test(data_file):
    ds = DataSet.from_csv_file(data_file)
    
    print("Ordinally encoding data")
    cols = list(ds.data.columns)
    for c in cols:
        ds.ordinal_encode(c, inplace=True)
    
    print(ds.summary())
    print()
    print("5 Random Samples")
    print(ds.data.sample(5))
    
def run_computer_hardware_test(data_file):
    ds = DataSet.from_csv_file(data_file)
    print(ds.summary())
    
def run_congressional_voting_records_test(data_file):
    ds = DataSet.from_csv_file(data_file)
    print(ds.summary())
    
def run_forest_fires_test(data_file):
    ds = DataSet.from_csv_file(data_file)
    print(ds.summary())
    
data_sets = {
    "Abalone": ("Abalone/abalone.data", run_abalone_tests),
    "Breast Cancer": ("Breast Cancer/breast-cancer-wisconsin.data", run_breast_cancer_test),
    "Car Evaluation": ("Car Evaluation/car.data", run_car_evaluation_test),
    "Computer Hardware": ("Computer Hardware/machine.data", run_computer_hardware_test),
    "Congressional Voting Records": ("Congressional Voting Records/house-votes-84.data", run_congressional_voting_records_test),
    "Forest Fires": ("Forest Fires/forestfires.data", run_forest_fires_test),
}

def run_tests():
    for dataset_name in data_sets:
        
        console_cols = os.get_terminal_size().columns
        chars_center = len(dataset_name)
        chars_ends = (console_cols - chars_center) // 2
        border_ends = "".join(["="] * chars_ends)        
        border_center = "".join(["="] * chars_center)    
        
        print(f"{border_ends}{dataset_name}{border_ends}")
        data_path = os.path.join(DATASETS_DIR, data_sets[dataset_name][0])
        test_fn = data_sets[dataset_name][1]
        test_fn(data_path)
        
        print(f"{border_ends}{border_center}{border_ends}")
        print()
        
        
if __name__ == "__main__":
    run_tests()