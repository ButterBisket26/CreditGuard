import os
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

def load_env():
    """Manually parse .env file to load database environment variables."""
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
    if os.path.exists(env_path):
        with open(env_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, val = line.split('=', 1)
                    os.environ[key.strip()] = val.strip()

# Load env variables immediately when module is imported
load_env()

def get_db_params():
    """Retrieve database parameters from environment variables."""
    return {
        "host": os.getenv("DB_HOST", "127.0.0.1"),
        "port": os.getenv("DB_PORT", "5432"),
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "dbname": os.getenv("DB_NAME", "loan_default")
    }

def get_connection(create_db_if_missing=True):
    """
    Establish a connection to the PostgreSQL database.
    If the target database does not exist and create_db_if_missing is True,
    it connects to the default 'postgres' database and creates the target database.
    """
    params = get_db_params()
    dbname = params["dbname"]
    
    try:
        # Attempt to connect to the target database
        conn = psycopg2.connect(**params)
        return conn
    except psycopg2.OperationalError as e:
        if create_db_if_missing and "does not exist" in str(e):
            print(f"Database '{dbname}' does not exist. Attempting to create it...")
            # Connect to default 'postgres' database to create the new one
            admin_params = params.copy()
            admin_params["dbname"] = "postgres"
            
            try:
                admin_conn = psycopg2.connect(**admin_params)
                admin_conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
                with admin_conn.cursor() as cursor:
                    # Sanitize the dbname by double quoting to prevent SQL issues
                    cursor.execute(f'CREATE DATABASE "{dbname}";')
                admin_conn.close()
                print(f"Database '{dbname}' created successfully.")
                
                # Retry connection to the newly created database
                return psycopg2.connect(**params)
            except Exception as create_err:
                raise RuntimeError(f"Failed to auto-create database '{dbname}': {create_err}") from create_err
        else:
            raise e

def setup_schema(schema_file_path=None):
    """Read schema SQL file and initialize the tables and indexes."""
    if schema_file_path is None:
        schema_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'sql', 'schema.sql'
        )
    
    print(f"Reading schema from {schema_file_path}...")
    with open(schema_file_path, 'r', encoding='utf-8') as f:
        schema_sql = f.read()
    
    print("Executing schema setup in database...")
    conn = get_connection()
    try:
        conn.autocommit = False
        with conn.cursor() as cursor:
            cursor.execute(schema_sql)
        conn.commit()
        print("Schema setup completed successfully.")
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Failed to execute schema SQL: {e}") from e
    finally:
        conn.close()

if __name__ == '__main__':
    # Running this file directly will test the connection and setup schema
    try:
        print("Testing database connection and setting up schema...")
        setup_schema()
    except Exception as err:
        print(f"Error during database initialization: {err}")
