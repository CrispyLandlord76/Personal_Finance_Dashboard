import sqlite3
import sys

def setup_database(db_path='Raw_data.db'):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()

    # Create Assets Table
    c.execute('''
    CREATE TABLE IF NOT EXISTS assets (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        type TEXT NOT NULL,
        nickname TEXT,
        amount REAL NOT NULL
    )
    ''')

    # Create Liabilities Table
    c.execute('''
    CREATE TABLE IF NOT EXISTS liabilities (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        year INTEGER NOT NULL,
        month INTEGER NOT NULL,
        type TEXT NOT NULL,
        nickname TEXT,
        amount REAL NOT NULL
    )
    ''')

    conn.commit()
    conn.close()

if __name__ == '__main__':
    db_path = sys.argv[1] if len(sys.argv) > 1 else 'Raw_data.db'
    setup_database(db_path)
