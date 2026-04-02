from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os
from pathlib import Path

Base = declarative_base()

class PollingData(Base):
    __tablename__ = 'polling_data'

    id = Column(Integer, primary_key=True)
    poll_id = Column(Integer)
    state = Column(String)
    sample_size = Column(Integer)
    candidate_name = Column(String)
    pct = Column(Float)
    party = Column(String)

def setup_database():
    connection_string = os.getenv("DB_CONNECTION_STRING")
    if connection_string is None:
        raise ValueError("DB_CONNECTION_STRING environment variable is not set!")
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    return engine

def save_data_to_db(cleaned_file_path, engine):
    data = pd.read_csv(cleaned_file_path)

    # Keep only columns that match the schema
    db_columns = ['poll_id', 'state', 'sample_size', 'candidate_name', 'pct', 'party']
    data = data[[col for col in db_columns if col in data.columns]]

    try:
        data.to_sql('polling_data', con=engine, if_exists='append', index=False)
        print("Cleaned polling data has been successfully loaded into the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    data_dir = base_dir / '..' / 'data' / 'polling_data'
    cleaned_file_path = data_dir / 'cleaned_polls.csv'

    engine = setup_database()
    save_data_to_db(cleaned_file_path, engine)
