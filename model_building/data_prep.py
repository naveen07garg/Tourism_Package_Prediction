#=== Import required libs ===========================================
# for data manipulation
import pandas as pd
import sklearn

# for creating a folder
import os

# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split

# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi
#=====================================================================

# Define constants for the dataset and output paths
data_dir="data/"
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/naveen07garg/Tourism-Package-Prediction/tourism.csv"
tour_data_raw = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

#=== Drop the unique identifier columns ===
tour_data_raw.drop(columns=['Unnamed: 0'], inplace=True)
tour_data_raw.drop(columns=['CustomerID'], inplace=True)

#=== Treatment of incorrect values ====================================

# Check how many records are with Gender Fe Male ===
print("Number of records with Gender as ''Fe Male'' :["+tour_data_raw.query('Gender == "Fe Male"')['Gender'].count().astype(str)+"]")

# correct the values
tour_data_treated=tour_data_raw.copy()
tour_data_treated.loc[tour_data_treated['Gender'] == 'Fe Male', 'Gender'] = 'Female'

#=======================================================================

# Encoding the categorical 'Type' column
#label_encoder = LabelEncoder()
#tour_data_raw['ProdTaken'] = label_encoder.fit_transform(tour_data_raw['ProdTaken'])

target_col = 'ProdTaken'

# Split into X (features) and y (target)
print("Split train and test dataset.\n")
X = tour_data_treated.drop(columns=[target_col])
y = tour_data_treated[target_col]

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Write train and test dataset to the file.\n")
Xtrain.to_csv(data_dir+"Xtrain.csv",index=False)
Xtest.to_csv(data_dir+"Xtest.csv",index=False)
ytrain.to_csv(data_dir+"ytrain.csv",index=False)
ytest.to_csv(data_dir+"ytest.csv",index=False)


files = [data_dir+"Xtrain.csv",data_dir+"Xtest.csv",data_dir+"ytrain.csv",data_dir+"ytest.csv"]

print("Upload train and test dataset file to Hugging Face.\n")
for file_path in files:
    print("\nUploading file - [ "+ file_path + " ]")
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],  # just the filename
        repo_id="naveen07garg/Tourism-Package-Prediction",
        repo_type="dataset",
    )
