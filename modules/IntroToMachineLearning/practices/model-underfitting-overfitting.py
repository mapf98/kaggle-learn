import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return mae

def get_best_mae(current, candidate):
    if current['mae'] == 0 or current['mae'] >= candidate['mae']:
        return candidate
    else:
        return current
    
def get_worst_mae(current, candidate):
    if current['mae'] == 0 or current['mae'] <= candidate['mae']:
        return candidate
    else:
        return current

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

best_mae_case = {'leaf_nodes': 0, 'mae': 0}
worst_mae_case = {'leaf_nodes': 0, 'mae': 0}

# Compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [2, 5, 50, 100, 500, 1000, 5000, 5500, 10000, 30000]:
    # Calculate MAE
    candidate_mae = get_mae(max_leaf_nodes, train_data_x, validation_data_x, train_data_y, validation_data_y)

    # Find best and worst MAE
    best_mae_case = get_best_mae(best_mae_case, {'leaf_nodes': max_leaf_nodes, 'mae': candidate_mae})
    worst_mae_case = get_worst_mae(worst_mae_case, {'leaf_nodes': max_leaf_nodes, 'mae': candidate_mae})

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, candidate_mae))

print()
print(f'Best case: {best_mae_case['leaf_nodes']} max leaf nodes with {best_mae_case['mae']} of MAE')
print(f'Worst case: {worst_mae_case['leaf_nodes']} max leaf nodes with {worst_mae_case['mae']} of MAE')