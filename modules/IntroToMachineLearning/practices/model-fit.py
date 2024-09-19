import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

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
start_date = datetime.now()
print(f'Model fitting begin: {start_date}')
model.fit(x, y)
print(f'Model fitting end: {datetime.now()}')
print(f'Model fitting duration: {datetime.now()-start_date}')