import pandas as pd
from sklearn.tree import DecisionTreeRegressor

melbourne_file_path = './modules/IntroToMachineLearning/data/melbourne_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

# Target
y = melbourne_data.Price

# Features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# Data filtered by features
x = melbourne_data[melbourne_features]

# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fit model
model.fit(x, y)

# Head predictions
print("Making predictions for the following 5 houses:")
print(x.head())

print("\nThe predictions are:")
head_predictions = model.predict(x.head())
print(head_predictions)