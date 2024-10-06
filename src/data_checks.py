import pandas as pd

# Load the dataset (assuming it's in the /data/ folder)
data = pd.read_csv('ep/Election-Prediction/data/polling_data/raw_polls.csv')

# Step 1: Check Historical Coverage
data['modeldate'] = pd.to_datetime(data['modeldate'], errors='coerce')
data['year'] = data['modeldate'].dt.year
polls_per_year = data['year'].value_counts().sort_index()
print("Polls per year:")
print(polls_per_year)

# Step 2: Check for missing values in key columns
important_columns = ['candidate_name', 'pct_estimate', 'modeldate', 'state']
missing_values = data[important_columns].isnull().sum()
print("\nMissing values in key columns:")
print(missing_values)

# Step 3: Group the data by year and count the number of polls
polls_by_year = data.groupby('year')['pct_estimate'].count()
print("\nNumber of polls per year:")
print(polls_by_year)

# Step 4: Descriptive statistics for polling percentages
polling_stats = data['pct_estimate'].describe()
print("\nPolling percentage statistics:")
print(polling_stats)
