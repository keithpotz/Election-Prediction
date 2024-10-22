import pandas as pd
import os

# Load the dataset (assuming it's in the /data/ folder)
base_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(base_dir, '../data/polling_data/raw_polls.csv')

# Load the data
data = pd.read_csv(file_path, low_memory=False)

# Step 1: Check Historical Coverage (we'll use 'end_date' instead of 'modeldate')
if 'end_date' in data.columns:
    data['end_date'] = pd.to_datetime(data['end_date'], errors='coerce')
    data['year'] = data['end_date'].dt.year
    polls_per_year = data['year'].value_counts().sort_index()
    print("Polls per year:")
    print(polls_per_year)
else:
    print("'end_date' column not found, skipping historical coverage check.")

# Step 2: Check for missing values in key columns
important_columns = ['candidate_name', 'pct', 'state']
missing_values = data[important_columns].isnull().sum()
print("\nMissing values in key columns:")
print(missing_values)

# Step 3: Group the data by year and count the number of polls
if 'year' in data.columns:
    polls_by_year = data.groupby('year')['pct'].count()
    print("\nNumber of polls per year:")
    print(polls_by_year)
else:
    print("Year data not available, skipping group by year check.")

# Step 4: Descriptive statistics for polling percentages
polling_stats = data['pct'].describe()
print("\nPolling percentage statistics:")
print(polling_stats)
