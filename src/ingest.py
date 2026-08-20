import os
import time
from db import get_connection, setup_schema

def ingest_data(csv_file_path=None):
    """
    Ingests the cleaned loans CSV file into the PostgreSQL 'loans' table.
    Uses the native PostgreSQL COPY command for maximum performance.
    """
    if csv_file_path is None:
        csv_file_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), 'data', 'loans_clean.csv'
        )
        
    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"Cleaned CSV file not found at: {csv_file_path}. Please run notebooks/01_explore.ipynb first.")
        
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            # Check if the table already exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM pg_tables 
                    WHERE schemaname = 'public' 
                    AND tablename = 'loans'
                );
            """)
            table_exists = cursor.fetchone()[0]
            
            if not table_exists:
                print("Table 'loans' does not exist in the database. Initializing schema...")
                # Close this connection before running setup_schema which opens its own connection
                cursor.close()
                conn.close()
                
                setup_schema()
                
                # Re-open connection
                conn = get_connection()
            else:
                print("Table 'loans' already exists. Truncating table to ensure a clean reload...")
                cursor.execute("TRUNCATE TABLE loans RESTART IDENTITY;")
                conn.commit()
                print("Table truncated and identity counter reset.")

        # PostgreSQL COPY command configuration
        # Column names must match the columns in the CSV headers in order
        columns = (
            "loan_amnt", "term", "int_rate", "grade", "sub_grade", 
            "emp_length_years", "emp_length_missing", "home_ownership", 
            "annual_inc", "purpose", "dti", "dti_missing", "delinq_2yrs", 
            "open_acc", "pub_rec", "revol_bal", "revol_util", "revol_util_missing", 
            "total_acc", "issue_d", "issue_year", "issue_quarter", 
            "addr_state", "application_type", "default_flag"
        )
        col_str = ", ".join(columns)
        
        # NULL '' treats empty fields in CSV (e.g. ,, or ,"",) as NULL in PostgreSQL
        copy_sql = f"""
            COPY loans ({col_str})
            FROM STDIN
            WITH (FORMAT CSV, HEADER TRUE, NULL '');
        """
        
        print("Streaming data to PostgreSQL via COPY command...")
        start_time = time.time()
        
        with open(csv_file_path, 'r', encoding='utf-8') as f:
            with conn.cursor() as cursor:
                cursor.copy_expert(copy_sql, f)
                
        conn.commit()
        duration = time.time() - start_time
        print(f"Data ingestion completed in {duration:.2f} seconds!")
        
        # Verify the row count in the database
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM loans;")
            count = cursor.fetchone()[0]
            print(f"Success! Total rows in table 'loans': {count:,}")
            
    except Exception as e:
        if conn:
            conn.rollback()
        raise RuntimeError(f"Data ingestion failed: {e}") from e
    finally:
        if conn:
            conn.close()

if __name__ == '__main__':
    try:
        ingest_data()
    except Exception as err:
        print(f"\nError: {err}")
