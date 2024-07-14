import sqlite3

def insert_asset(year, month, type, nickname, amount):
    conn = sqlite3.connect('finance.db')
    c = conn.cursor()
    c.execute('INSERT INTO assets (year, month, type, nickname, amount) VALUES (?, ?, ?, ?)', (year, month, type, nickname, amount))
    conn.commit()
    conn.close()

def insert_liability(year, month, type, nickname, amount):
    conn = sqlite3.connect('finance.db')
    c = conn.cursor()
    c.execute('INSERT INTO liabilities (year, month, type, nickname, amount) VALUES (?, ?, ?, ?)', (year, month, type, nickname, amount))
    conn.commit()
    conn.close()

    # Example usage
#    if __name__ == '__main__':
#        insert_asset(2024, 6, 'Checking Account', 250)
#        insert_liability(2024, 6, 'Credit Card Payment', 500)
