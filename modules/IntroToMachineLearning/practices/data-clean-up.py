import pandas as pd

# save filepath to variable for easier access
melbourne_file_path = './modules/IntroToMachineLearning/data/melbourne_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

# dropna drops missing values (think of na as "not available") 
clean_melbourne_data = melbourne_data.dropna(axis=0)

print(f'Count before drop missing values: \n{melbourne_data.count(numeric_only=True)}')
print(f'\nCount after drop missing values: \n{clean_melbourne_data.count(numeric_only=True)}')