# Election Prediction Project

**Status**: 🚧 Work in Progress

This project aims to build a non-partisan, data-driven election prediction model using historical polling data. The goal is to use data from sources like FiveThirtyEight to train machine learning models that can predict election outcomes, starting with the 2024 U.S. presidential election.

## Things to do still:
- [x] Experiment with more complex machine learning models
- [x] Implement  election_model.py
- [x] fininsh modeling.py
- [x] Implement cross-validation and other evaluation techniques
- [x] Create a user interface to visulize predictions
- [] Develop documentation and contribution guidelines.

## Project Overview

The project is broken down into several key steps:

1. **Data Collection**: 
   - I used polling data from FiveThirtyEight ( You can find that data yourself [Here](https://github.com/fivethirtyeight/data/tree/master/polls)) to collect historical polling information for different elections. (NOTE: I will be adding more resources as time goes on. FiveThirtyEight is just a jumping off point.)
   - The raw data is stored in a CSV file, which is then cleaned and processed.

2. **Data Cleaning**: 
   - Unnecessary columns are removed, missing values are handled, and data types are converted.
   - The cleaned data is stored in a separate CSV file for use in model training.

3. **Database Integration**: 
   - The cleaned data is loaded into a PostgreSQL database using SQLAlchemy for easy querying and management.

4. **Model Training**: 
   - Data is extracted from the database and preprocessed for model training.
   - The model is trained using various machine learning algorithms (e.g., Linear Regression) to make predictions.
   - The model's performance is evaluated using metrics like Mean Squared Error (MSE) and R-squared.

5. **Future Enhancements** (To Do):
   - Implement other models like Decision Trees, Random Forests, or Neural Networks.
   - Expand the dataset to include more features such as demographics, economic indicators, and more.
   - Create a visual interface for users to explore predictions.
   - Implement more robust evaluation and cross-validation techniques.

## Prerequisites

Before running the project, make sure you have the following installed:

- Python 3.8+
- PostgreSQL
- Required Python packages (install using `pip install -r requirements.txt`):
  - `pandas`
  - `sqlalchemy`
  - `psycopg2`
  - `scikit-learn`
  - `joblib`
  - `matplotlib`
  - `numpy`
  - `fastapi`
  - `uvicorn`
  - `streamlit`


## Project Structure

```plaintext
project-directory/
│
├── data/
│   ├── raw/               # Directory for raw data files
│   ├── cleaned/           # Directory for cleaned data files
│   └── polling_data.csv   # Example of raw polling data file
│
├── src/
│   ├── data_collection.py # Script to collect data from APIs and sources
│   ├── data_cleaning.py   # Script to clean the data
│   ├── database.py        # Script to load data into the database
│   ├── modeling.py        # Script to train and evaluate the model
│   └── __init__.py        # Initialize the src module
│
└── README.md              # Project README file

 ```
 ## Setup
### Step 1: clone the repository

```bash
git clone https://github.com/keithpotz/Election-Perdiction.git
```
### Step 2 Install required Python packages in your directory

```bash
pip install -r requirements.txt
```

### Step 3: Setup your PostgreSQL:
Create Database and then connect in config.py
```python
export DB_CONNECTION_STRING='postgresql://username:password@localhost:5432/your_database_name'
```
#### **Note:*** 
Make sure that you change your USERNAME and PASSWORD and the DATABASE_NAME to what you have setup on your machine.

## Then run the scripts:

### Data Collection:
```bash
python src/data_collection.py
```
### Data Cleaning:
```bash
python src/data_cleaning.py
```

### Load Data into Database

```bash
python src/database.py
```

### Train the model

```bash
python src/modeling.py
```

---

## Current Limitations

#### Incomplete Dataset: 
The project currently uses historical polling data up to the 2020 election. Further data collection and preprocessing are required to improve prediction accuracy.

#### Basic Model:

The model currently uses simple linear regression. Future versions will explore more complex models and feature engineering techniques.



## Contributions:

This project is open source and contributions are welcome! Please feel free to fork the repository, make improvements, and submit a pull request.
