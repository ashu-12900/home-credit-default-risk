import pickle
import pandas as pd
from merge import load_and_merge
from data_pre import clean_data,feature_engineering
features_path=r"C:\Users\PC\OneDrive\Desktop\work\home-credit-default-risk\features.pkl"
output_path=r"C:\Users\PC\OneDrive\Desktop\work\home-credit-default-risk\test_merged.parquet"
def prep_test_data():
    with open(features_path, "rb") as f:
        selected_features = pickle.load(f)
    df = load_and_merge("application_test.csv")
    df = feature_engineering(df)
    df = clean_data(df) 
    if "SK_ID_CURR" not in selected_features:
        selected_features=["SK_ID_CURR"] + selected_features
    df=df.reindex(columns=selected_features,fill_value=0)
    df.to_parquet(output_path,index=False)
    print("saved:",df.shape)
if __name__ == "__main__":
    prep_test_data()   
