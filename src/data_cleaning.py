import pandas as pd
import os

def load_and_merge(base_path, supplement_path=None):
    """Load base dataset and optionally merge a supplemental file, normalizing columns."""
    def normalize(df):
        if 'candidatevotes' in df.columns and 'votes' not in df.columns:
            df = df.rename(columns={'candidatevotes': 'votes'})
        return df

    base = normalize(pd.read_csv(base_path, low_memory=False))

    if supplement_path and os.path.exists(supplement_path):
        supp = normalize(pd.read_csv(supplement_path, low_memory=False))
        # Remove any overlapping years from base before merging
        if 'year' in supp.columns and 'year' in base.columns:
            overlap_years = supp['year'].unique()
            base = base[~base['year'].isin(overlap_years)]
        base = pd.concat([base, supp], ignore_index=True)

    return base

def clean_polling_data(file_path, supplement_path=None):
    data = load_and_merge(file_path, supplement_path)

    # 2. Filter to general elections only, exclude write-ins
    # Keep rows where stage is GEN or where stage is missing (older datasets don't have this column)
    if 'stage' in data.columns:
        data = data[(data['stage'] == 'GEN') | (data['stage'].isna())]
    if 'writein' in data.columns:
        data = data[data['writein'].astype(str).str.upper() == 'FALSE']

    # 3. Drop rows where totalvotes is 0 or missing (prevents division by zero)
    data = data[pd.to_numeric(data['totalvotes'], errors='coerce').fillna(0) > 0]

    # 4. Compute vote percentage from raw vote counts
    data['pct'] = (data['votes'] / data['totalvotes'] * 100).round(2)

    # 5. Rename columns to match the rest of the pipeline
    data = data.rename(columns={
        'candidate': 'candidate_name',
        'party_simplified': 'party',
        'totalvotes': 'sample_size',
    })

    # 5. Add a poll_id from the row index
    data = data.reset_index(drop=True)
    data['poll_id'] = data.index + 1

    # 6. Keep only columns useful for modeling
    keep_columns = ['poll_id', 'year', 'state', 'state_po', 'candidate_name', 'party', 'votes', 'pct', 'sample_size']
    data = data[keep_columns]

    # 7. Handle missing values
    data = data.dropna(subset=['candidate_name', 'pct'])
    data['state'] = data['state'].fillna('Unknown')
    data = data[data['state'] != 'Unknown']
    data = data.fillna(0)

    # 8. Save cleaned data
    base_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(base_dir, '../src/data/polling_data')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cleaned_polls.csv')
    data.to_csv(save_path, index=False)

    return data

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path        = os.path.join(base_dir, 'data/polling_data/complete_data.csv')
    supplement_path  = os.path.join(base_dir, 'data/polling_data/2024president.csv')
    cleaned_data = clean_polling_data(file_path, supplement_path)
    print(f"Cleaned data saved to src/data/polling_data/cleaned_polls.csv")
    print(f"Shape: {cleaned_data.shape}")
    print(cleaned_data.head())
