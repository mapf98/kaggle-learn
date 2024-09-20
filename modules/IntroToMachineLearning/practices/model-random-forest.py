import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from datetime import datetime

melbourne_file_path = './modules/IntroToMachineLearning/data/melbourne_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
melbourne_data = melbourne_data.dropna(axis=0)

def get_mae_with_decision_tree(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

# Target
y = melbourne_data.Price

# Features
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']

# Data filtered by features
x = melbourne_data[melbourne_features]

# Split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to the random_state argument guarantees we get the same split every time we run this script.
train_data_x, validation_data_x, train_data_y, validation_data_y = train_test_split(x, y, random_state = 0)

# Define Random Forest model. Specify a number for random_state to ensure same results each run
model = RandomForestRegressor(random_state=1)

# Fit model
start_date = datetime.now()
print(f'Model fitting begin: {start_date}')
model.fit(train_data_x, train_data_y)
print(f'Model fitting end: {datetime.now()}')
print(f'Model fitting duration: {datetime.now()-start_date}\n')

# Make predictions
predicted_home_prices = model.predict(validation_data_x)

# Calculate MAE with RandomForestRegressor
mae = mean_absolute_error(validation_data_y, predicted_home_prices)
print(f'Mean absolute error on predicted data with RandomForestRegressor: {mae} USD')
print(f'On average, our predictions are off by about {mae} USD')

print()

# Calculate MAE with DecisionTreeRegressor
# With 500 max_leaf_nodes wich is the best option previouly calculated on model-underfitting-overfitting.py
mae_with_decision_tree = get_mae_with_decision_tree(500, train_data_x, validation_data_x, train_data_y, validation_data_y )
print(f'Mean absolute error on predicted data with DecisionTreeRegressor: {mae_with_decision_tree} USD')
print(f'On average, our predictions are off by about {mae_with_decision_tree} USD')

print()

print(f'The RandomForestRegressor is more accurate than DecisionTreeRegressor by {mae_with_decision_tree-mae} USD')
print(f'This means that the mean absolute error with RandomForestRegressor is less and the predictions are more accurate')