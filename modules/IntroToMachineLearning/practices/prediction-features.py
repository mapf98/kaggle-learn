import pandas as pd

melbourne_file_path = './modules/IntroToMachineLearning/data/melbourne_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

# Prediction target (usually called "y")
y = melbourne_data.Price # This data is obtained from the columns of the data frame

# The features are the columns from that we want to make a prediction, in this case, based on these features we want to predict the price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Features (usually called "x")
x = melbourne_data[melbourne_features]

print(f'Data filtered by prediction features:\n{x}')
print(f'\nFirst five rows from filtered data:\n{x.head()}')
print(f'\nFilterd data description:\n{x.describe()}')