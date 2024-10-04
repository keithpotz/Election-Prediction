from sqlalchemy import create_engine, Column, Integer, String, Float, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os

# Define the base model for the polling data table
Base = declarative_base()

class PollingData(Base):
    __tablename__ = 'polling_data'
    
    id = Column(Integer, primary_key=True)
    poll_id = Column(Integer)
    state = Column(String)
    sample_size = Column(Integer)
    population = Column(String)
    candidate_name = Column(String)
    pct = Column(Float)
    party = Column(String)

def setup_database():

    # Set up the database connection using SQLAlchemy
    connection_string = os.getenv("DB_CONNECTION_STRING")
    if connection_string is None:
        raise ValueError("DB_CONNECTION_STRING enivronment variable is not set!")
    engine = create_engine(connection_string)
    Base.metadata.create_all(engine)
    return engine

def save_data_to_db(cleaned_file_path, engine):
    # Load cleaned data from CSV
    data = pd.read_csv(cleaned_file_path)

    # Create a database session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Iterate over rows and add them to the session
    for index, row in data.iterrows():
        poll = PollingData(
            poll_id=row['poll_id'],
            state=row['state'],
            sample_size=row['sample_size'],
            population=row['population'],
            candidate_name=row['candidate_name'],
            pct=row['pct'],
            party=row['party']
        )
        session.add(poll)

    # Commit the session to the database
    session.commit()

# Main function to call database setup and save data
if __name__ == "__main__":
    cleaned_file_path = '../ep/data/polling_data/cleaned_polls.csv'  # Path to your cleaned data
    engine = setup_database()  # Set up the connection to the database
    save_data_to_db(cleaned_file_path, engine)  # Save the cleaned data into the database
    print("Cleaned polling data has been successfully loaded into the database.")
