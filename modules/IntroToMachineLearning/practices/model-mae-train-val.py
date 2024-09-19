import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

melbourne_file_path = './modules/IntroToMachineLearning/data/melbourne_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

# Target
y = melbourne_data.Price

# Features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

# Data filtered by features
x = melbourne_data[melbourne_features]

# Split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script.
train_data_x, validation_data_x, train_data_y, validation_data_y = train_test_split(x, y, random_state = 0)

# Define model. Specify a number for random_state to ensure same results each run
model = DecisionTreeRegressor(random_state=1)

# Fit the model with train data
start_date = datetime.now()
print(f'Model fitting with train data begin: {start_date}')
model.fit(train_data_x, train_data_y)
print(f'Model fitting with train data end: {datetime.now()}')
print(f'Model fitting with train data duration: {datetime.now()-start_date}\n')

# Predictions
predictions_with_validation_data = model.predict(validation_data_x)

# Mean absolute error with train and validation data
mae = mean_absolute_error(validation_data_y, predictions_with_validation_data)
print(f'Mean absolute error on predicted data: {mae} USD')
print(f'On average, our predictions are off by about {mae} USD')