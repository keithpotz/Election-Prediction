from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.orm import declarative_base, sessionmaker
import pandas as pd
import os
from pathlib import Path
from config import DB_CONNECTION_STRING

Base = declarative_base()

class PollingData(Base):
    __tablename__ = 'polling_data'

    id = Column(Integer, primary_key=True)
    poll_id = Column(Integer)
    year = Column(Integer)
    state = Column(String)
    candidate_name = Column(String)
    party = Column(String)
    votes = Column(Integer)
    pct = Column(Float)
    sample_size = Column(Integer)

def setup_database():
    engine = create_engine(DB_CONNECTION_STRING)
    Base.metadata.create_all(engine)
    return engine

def save_data_to_db(cleaned_file_path, engine):
    data = pd.read_csv(cleaned_file_path)

    db_columns = ['poll_id', 'year', 'state', 'candidate_name', 'party', 'votes', 'pct', 'sample_size']
    data = data[[col for col in db_columns if col in data.columns]]

    try:
        data.to_sql('polling_data', con=engine, if_exists='append', index=False)
        print("Cleaned election data has been successfully loaded into the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent
    cleaned_file_path = base_dir / '../src/data/polling_data/cleaned_polls.csv'

    engine = setup_database()
    save_data_to_db(cleaned_file_path, engine)
