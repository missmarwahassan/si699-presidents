# pip install kagglehub[pandas-datasets]

import os
import pandas as pd

import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
file_path1 = "presidential_speeches_full.csv"
file_path2 = "documents.pkl"

# Load the latest version
PRESIDENTIAL_SPEECHES = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "vaishven/presidential-spoken-addresses-and-remarks",
  file_path1,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

path = kagglehub.dataset_download("vaishven/presidential-spoken-addresses-and-remarks")

pkl_path = os.path.join(path, file_path2)

DOCUMENTS = pd.read_pickle(pkl_path)
print("Loaded chunked DOCUMENTS:")
print(DOCUMENTS.head())

print("First 5 records:", PRESIDENTIAL_SPEECHES.head())