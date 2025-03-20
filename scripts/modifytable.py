import sqlite3
import os

# Path to your SQLite database
db_path = './.vv8.db'

def modify_table():
    # Check if the database exists
    if not os.path.exists(db_path):
        print("Database does not exist. Exiting.")
        return
    
    # Connect to the SQLite database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop the 'submissions' table if it exists and recreate it
    try:
        # Drop the existing table if it exists
        cursor.execute('DROP TABLE IF EXISTS submissions')

        # Create the new 'submissions' table with the 'task_id' column
        cursor.execute('''
            CREATE TABLE submissions (
                submission_id TEXT NOT NULL PRIMARY KEY,
                task_id TEXT NOT NULL,
                url TEXT NOT NULL,
                start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scan_domain TEXT,
                actions JSON
            )
        ''')

        # Commit changes
        conn.commit()
        print("Table 'submissions' has been updated successfully.")
    
    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    finally:
        # Close the connection
        conn.close()

# Run the function to modify the table
modify_table()
