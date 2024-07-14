import sqlite3

def read_assets():
    conn = sqlite3.connect('Raw_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM assets')
    rows = c.fetchall()
    conn.close()
    return rows

def read_liabilities():
    conn = sqlite3.connect('Raw_data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM liabilities')
    rows = c.fetchall()
    conn.close()
    return rows

# Example usage
if __name__ == '__main__':
    assets = read_assets()
    liabilities = read_liabilities()

    print("Assets:")
    for row in assets:
        print(row)

    print("\nLiabilities:")
    for row in liabilities:
        print(row)
