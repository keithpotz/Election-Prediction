import requests
import pandas as pd
import os

# Function to fetch polling data from FiveThirtyEight's GitHub
def fetch_polling_data():
    url = "https://projects.fivethirtyeight.com/polls-page/data/president_primary_polls.csv"  # Replace with the correct CSV link
    response = requests.get(url)

    if response.status_code == 200:
        data = pd.read_csv(url, on_bad_lines='skip',delimiter=',', low_memory=False)
        return data
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None
#main function to call the fetch_polling_data function
if __name__ == "__main__":
    polling_data = fetch_polling_data()
    if polling_data is not None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        save_path = os.path.join(base_dir, '../src/data/polling_data/raw_polls.csv')

        polling_data.to_csv(save_path, index=False)


        print("Polling data successfully saved.")
