import os

DB_CONNECTION_STRING = os.getenv("DB_CONNECTION_STRING")
if DB_CONNECTION_STRING is None:
    raise ValueError("DB_CONNECTION_STRING environment variable is not set!")
