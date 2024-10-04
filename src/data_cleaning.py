import pandas as pd
import os

def clean_polling_data(file_path):
    # Load the data
    data = pd.read_csv(file_path, low_memory=False)
    
    # 1. Drop irrelevant columns
    irrelevant_columns = ['timestamp', 'comment', 'sponsor_ids', 'pollster_rating_id', 'notes', 'url','pollster_id','pollster','sponsors','display_name','pollster_name','pollster_rating_name','ranked_choice_round',
    'sponsor_candidate_id','sponsor_candidate','sponsor_candidate_party','endorsed_candidate_id','endorsed_candidate_name','endorsed_candidate_party','url_article',
    'url_topline','url_crosstab','source','internal','partisan','poll_start_date','poll_end_date',]  
    # Add the actual irrelevant columns from the dataset
    data.drop(columns=irrelevant_columns, inplace=True, errors='ignore')
    
    # 2. Handle missing values
    # Drop rows with missing values in key columns like 'candidate_name' and 'pct'
    data.dropna(subset=['candidate_name', 'pct'], inplace=True)

    # Optionally fill remaining missing values with 0
    data.fillna(0, inplace=True)

    # 3. Rename columns
    rename_columns = {
        'start_date': 'poll_start_date',  # Replace with relevant column renames if necessary
        'end_date': 'poll_end_date'
    }
    data.rename(columns=rename_columns, inplace=True)
   
    # 4. Convert date columns to datetime format
    date_columns = ['poll_start_date', 'poll_end_date']
    for col in date_columns:
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], format='%y-%m-%d', errors='coerce') # convert invalid dates like '0' to NaT

    # Save the cleaned data
    os.makedirs('../ep/data/polling_data', exist_ok=True)
    data.to_csv('../ep/data/polling_data/cleaned_polls.csv', index=False)

    return data

# Main function to call the clean_polling_data function
if __name__ == "__main__":
    file_path = '../ep/data/polling_data/raw_polls.csv'  # Replace with the path to your raw data file
    cleaned_data = clean_polling_data(file_path)
    print("Polling data cleaned and saved to ../ep/data/polling_data/cleaned_polls.csv")
    print(cleaned_data.head())