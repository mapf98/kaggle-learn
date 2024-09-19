import pandas as pd

melbourne_file_path = './modules/IntroToMachineLearning/data/melbourne_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

# Prediction target (usually called "y")
y = melbourne_data.Price # This data is obtained from the columns of the data frame

print(f'Prediction target:\n{y}')