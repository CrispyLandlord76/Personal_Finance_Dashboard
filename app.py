import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from fpdf import FPDF
import os
import sys
import subprocess

# App name and icon
st.set_page_config(page_title="Personal Finance Dashboard", page_icon="💰")

# Check if database exists, if not, run setup_db.py
db_path = 'Raw_data.db'
if not os.path.exists(db_path):
    setup_db_script = os.path.join(os.path.dirname(__file__), 'setup_db.py')
    result = subprocess.run([sys.executable, setup_db_script], capture_output=True, text=True)
    if result.returncode != 0:
        st.error(f"Error setting up the database:\n{result.stderr}")
    else:
        st.success("Database setup complete.")

# Initialize session state for goals if not already present
if 'goals' not in st.session_state:
    st.session_state.goals = []

# Load custom CSS
def local_css(file_name):
    if os.path.exists(file_name):
        with open(file_name) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    else:
        st.warning(f"Custom CSS file '{file_name}' not found.")

local_css("style.CSS")

# Get the correct path for the database file
if hasattr(sys, '_MEIPASS'):
    db_path = os.path.join(sys._MEIPASS, 'Raw_data.db')
else:
    db_path = os.path.join(os.path.dirname(__file__), 'Raw_data.db')

def create_connection():
    """ create a database connection to the SQLite database specified by the db_path """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        st.error(e)
    return conn

# Import database
def import_database(uploaded_file):
    if uploaded_file is not None:
        with open(db_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        st.success("Database imported successfully!")

# Function to insert an asset into the database
def insert_asset(year, month, type, nickname, amount):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO assets (year, month, type, nickname, amount) VALUES (?, ?, ?, ?, ?)', (year, month, type, nickname, amount))
    conn.commit()
    conn.close()

# Function to insert a liability into the database
def insert_liability(year, month, type, nickname, amount):
    conn = create_connection()
    c = conn.cursor()
    c.execute('INSERT INTO liabilities (year, month, type, nickname, amount) VALUES (?, ?, ?, ?, ?)', (year, month, type, nickname, amount))
    conn.commit()
    conn.close()

# Function to read assets from the database
def read_assets():
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM assets')
    rows = c.fetchall()
    conn.close()
    return rows

# Function to read liabilities from the database
def read_liabilities():
    conn = create_connection()
    c = conn.cursor()
    c.execute('SELECT * FROM liabilities')
    rows = c.fetchall()
    conn.close()
    return rows

# Function to delete an asset from the database
def delete_asset(asset_id):
    conn = create_connection()
    c = conn.cursor()
    c.execute('DELETE FROM assets WHERE id = ?', (asset_id,))
    conn.commit()
    conn.close()

# Function to delete a liability from the database
def delete_liability(liability_id):
    conn = create_connection()
    c = conn.cursor()
    c.execute('DELETE FROM liabilities WHERE id = ?', (liability_id,))
    conn.commit()
    conn.close()

# Function to format unique data for the dropdown menu
def format_unique_data_for_selection(data):
    unique_data = {f"{row[3]} - {row[4]}" if row[4] else row[3] for row in data}
    return list(unique_data)

# Function to add a new goal
def add_goal(name, target, assets):
    st.session_state.goals.append({
        'name': name,
        'target': target,
        'assets': assets,
        'hidden': False,  # Add a hidden property to manage folding
        'show_trend': False  # Add a property to manage showing trend
    })
    save_goals()


# Function to display and track goals
def display_goals(asset_values):
    currency_symbol = currencies[st.session_state.currency]
    if st.session_state.goals:
        for idx, goal in enumerate(st.session_state.goals):
            if 'show_trend' not in goal:
                goal['show_trend'] = False
            if goal['hidden']:
                st.write(f"### {goal['name']}")
                if st.button(f"Show", key=f"show_{idx}"):
                    st.session_state.goals[idx]['hidden'] = False
                    save_goals()
                    st.experimental_rerun()
            else:
                st.write(f"### {goal['name']}")
                st.write(f"Target Amount: {goal['target']}")
                st.write(f"Included Assets: {', '.join(goal['assets'])}")
                current_amount = sum([asset_values[asset] for asset in goal['assets']])
                if current_amount >= goal['target']:
                    progress = 100
                    st.success("You have reached the saving goal!")
                else:
                    progress = (current_amount / goal['target']) * 100
                st.progress(progress / 100)
                st.write(f"Progress: {progress:.2f}% ({current_amount} / {goal['target']})") 
                col1, col2, col3 = st.columns(3)
                if col1.button(f"Hide", key=f"hide_{idx}"):
                    st.session_state.goals[idx]['hidden'] = True
                    save_goals()
                    st.experimental_rerun()
                if col2.button(f"Delete", key=f"delete_{idx}"):
                    del st.session_state.goals[idx]
                    save_goals()
                    st.experimental_rerun()
                if col3.button(f"Trend", key=f"trend_{idx}"):
                    st.session_state.goals[idx]['show_trend'] = not st.session_state.goals[idx]['show_trend']
                    save_goals()
                    st.experimental_rerun()
                if st.session_state.goals[idx]['show_trend']:
                    plot_trend(goal, asset_values, idx)
                   
    else:
        st.write("No goals added yet.")

# Function to save goals to a file
def save_goals():
    with open('goals.json', 'w') as f:
        json.dump(st.session_state.goals, f)

# Function to load goals from a file
def load_goals():
    try:
        with open('goals.json', 'r') as f:
            st.session_state.goals = json.load(f)
        for goal in st.session_state.goals:
            if 'show_trend' not in goal:
                goal['show_trend'] = False
    except FileNotFoundError:
        st.session_state.goals = []

# Function to plot data plotly line and bar chart combined
def plot_combined_chart(data, title):
    df = pd.DataFrame(data, columns=['Date', 'Amount'])
    df['MoM Change'] = df['Amount'].pct_change() * 100  # Calculate MoM percentage change
    df['MoM Change'] = df['MoM Change'].round(2)  # Limit digits to 2 decimal places

    currency_symbol = currencies[st.session_state.currency]

    fig = px.line(df, x='Date', y='Amount', title=title, markers=True,
                  labels={'Amount': f'Amount ({currency_symbol})'})

    bar = go.Bar(
        x=df['Date'],
        y=df['MoM Change'],
        name='MoM Change (%)',
        marker=dict(
            color=['green' if v > 0 else 'red' for v in df['MoM Change']],
            opacity=0.5  # Set bar transparency to 50%
        ),
        yaxis='y2'
    )

    fig.add_trace(bar)

    fig.update_traces(
        hovertemplate=f'Date: %{{x|%Y-%m}}<br>Amount: {currency_symbol}%{{y:.2f}}<extra></extra>',
        selector=dict(type='scatter')
    )

    fig.update_traces(
        hovertemplate=f'Date: %{{x|%Y-%m}}<br>MoM Change: %{{y:.2f}}%<extra></extra>',
        selector=dict(type='bar')
    )

    fig.update_layout(
        transition_duration=500,
        xaxis=dict(
            tickformat='%Y-%m',  # Format the x-axis to show only year and month
            dtick="M1"  # Set tick interval to 1 month
        ),
        yaxis=dict(title=f'Amount ({currency_symbol})'),
        yaxis2=dict(
            title='MoM Change (%)',
            overlaying='y',
            side='right',
            showgrid=False
        )
    )

    st.plotly_chart(fig, use_container_width=True)

#Function to plot pie chart
def plot_pie_chart(data, title):
    currency_symbol = currencies[st.session_state.currency]
    df = pd.DataFrame(data, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
    df['Label'] = df.apply(lambda row: f"{row['Type']} - {row['Nickname']}" if row['Nickname'] else row['Type'], axis=1)
    
    total_amount = df['Amount'].sum()
    fig = px.pie(df, names='Label', values='Amount', title=title)
    
    fig.update_traces(
        hovertemplate=f'%{{label}}<br>Amount: {currency_symbol}%{{value:.2f}}<extra></extra>'
    )
    fig.add_annotation(
        text=f"Total: {currency_symbol}{total_amount:.2f}",
        x=0.5, y=-0.1,
        showarrow=False,
        font=dict(size=14),
        xref="paper",
        yref="paper"
    )
    st.plotly_chart(fig, use_container_width=True)


#Define Net Worth Plot Function
def plot_net_worth(assets, liabilities, title):
    assets_df = pd.DataFrame(assets, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
    assets_df['Date'] = pd.to_datetime(assets_df[['Year', 'Month']].assign(DAY=1))
    assets_df = assets_df.groupby('Date')['Amount'].sum().reset_index()

    liabilities_df = pd.DataFrame(liabilities, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
    liabilities_df['Date'] = pd.to_datetime(liabilities_df[['Year', 'Month']].assign(DAY=1))
    liabilities_df = liabilities_df.groupby('Date')['Amount'].sum().reset_index()

    net_worth_df = pd.merge(assets_df, liabilities_df, on='Date', how='outer', suffixes=('_asset', '_liability')).fillna(0)
    net_worth_df['Net_Worth'] = net_worth_df['Amount_asset'] - net_worth_df['Amount_liability']

    net_worth_data = net_worth_df[['Date', 'Net_Worth']].copy()
    net_worth_data.rename(columns={'Net_Worth': 'Amount'}, inplace=True)

    plot_combined_chart(net_worth_data, title)

#Goal trend chase
def plot_trend(goal, asset_values, goal_idx):
    # Filter asset data for the selected assets
    assets = read_assets()
    selected_asset_data = [asset for asset in assets if (f"{asset[3]} - {asset[4]}" if asset[4] else asset[3]) in goal['assets']]

    # Aggregate data for plotting
    asset_df = pd.DataFrame(selected_asset_data, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
    asset_df['Date'] = pd.to_datetime(asset_df[['Year', 'Month']].assign(DAY=1))
    aggregated_asset_data = asset_df.groupby('Date')['Amount'].sum().reset_index()

    # Get the selected currency symbol
    #currency_symbol = st.session_state.currency
    currency_symbol = currencies[st.session_state.currency]
    # Create the bar plot
    fig = go.Figure()

    # Add goal amount bar
    goal_data = pd.DataFrame({
        'Date': pd.date_range(start=aggregated_asset_data['Date'].min(), end=aggregated_asset_data['Date'].max(), freq='MS'),
        'Target': goal['target']
    })
    fig.add_trace(go.Bar(
        x=goal_data['Date'], 
        y=goal_data['Target'], 
        name='Goal Amount', 
        opacity=0.5,
        hovertemplate=f'Target: {currency_symbol}%{{value:.2f}}<extra></extra>'
    ))

    # Add summary of selected assets
    fig.add_trace(go.Bar(
        x=aggregated_asset_data['Date'], 
        y=aggregated_asset_data['Amount'], 
        name='Assets Amount',
        hovertemplate=f'Amount: {currency_symbol}%{{y:.2f}}<extra></extra>'
    ))

    # Update layout
    fig.update_layout(
        title=f"Trend for {goal['name']}",
        xaxis_title="Date",
        yaxis_title=f'Amount ({currency_symbol})',
        barmode='overlay',
        hovermode="x unified"
    )

    st.plotly_chart(fig)

# Export database
def export_database():
    with open(db_path, 'rb') as f:
        st.download_button(
            label="Export Database",
            data=f,
            file_name='Raw_data.db',
            mime='application/octet-stream'
        )

# Function to export data to CSV
def export_to_csv(assets, liabilities, selected_months, start_date, end_date, currency_symbol):
    data = generate_balance_sheet_data(assets, liabilities, selected_months)
    if data is not None:
        # Add column headers and format amounts
        data.columns = ['Category'] + selected_months
        data = data.round(2)
        
        # Clean the Category column
        data['Category'] = data['Category'].apply(lambda x: '-'.join(filter(None, x.split(' - '))))
        
        filename = f"balance_sheet_{start_date.strftime('%Y-%m')}_to_{end_date.strftime('%Y-%m')}.csv"
        
        data.to_csv(filename, index=False)
        st.success(f"Data exported to CSV successfully with currency: {currency_symbol}!")
        st.download_button(
            label="Download CSV",
            data=data.to_csv(index=False).encode('utf-8'),
            file_name=filename,
            mime='text/csv',
        )
    else:
        st.warning("No data available for the selected date range.")

# Function to export data to PDF
class PDF(FPDF):
    # def header(self):
    #     self.set_font('Arial', 'B', 12)
    #     self.cell(0, 10, 'Balance Sheet', 0, 1, 'C')

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 7.5, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 7.5, body)
        self.ln()

def export_to_pdf(assets, liabilities, selected_months, start_date, end_date, memo, currency_symbol):
    data = generate_balance_sheet_data(assets, liabilities, selected_months)
    if data is not None:
        data.columns = ['Category'] + selected_months
        data = data.round(2)
        
        # Clean the Category column
        data['Category'] = data['Category'].apply(lambda x: '-'.join(filter(None, x.split(' - '))))
        
        filename = f"balance_sheet_{start_date.strftime('%Y-%m')}_to_{end_date.strftime('%Y-%m')}.pdf"
        
        pdf = PDF('L')  # Landscape orientation
        pdf.add_page()
        pdf.set_font('Arial', 'B', 8)
        pdf.cell(0, 7.5, 'Balance Sheet', 0, 1, 'C')
        pdf.ln(5)
        pdf.set_font('Arial', '', 8)
        pdf.cell(0, 7.5, f"Date Range: {start_date.strftime('%Y-%m')} to {end_date.strftime('%Y-%m')}", 0, 1, 'C')
        if memo:
            pdf.cell(0, 7.5, f"Memo: {memo}", 0, 1, 'C')
        pdf.ln(10)

        # Split data into chunks of 12 months
        chunk_size = 12
        num_chunks = (len(selected_months) + chunk_size - 1) // chunk_size
        chunks = [selected_months[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

        for chunk in chunks:
            pdf.set_font('Arial', 'B', 8)
            col_widths = [max(pdf.get_string_width(str(value)) for value in data[['Category'] + chunk].values[:, col]) + 5 for col in range(len(chunk) + 1)]
            
            # Header row
            for col, width in zip(['Category'] + chunk, col_widths):
                pdf.cell(width, 7.5, col, 1)
            pdf.ln()

            # Content rows
            pdf.set_font('Arial', '', 8)
            for row in data[['Category'] + chunk].itertuples():
                for idx, value in enumerate(row[1:]):
                    if idx == 0:  # First column is 'Category'
                        pdf.cell(col_widths[idx], 7.5, str(value), 1)
                    else:
                        pdf.cell(col_widths[idx], 7.5, f"{currency_symbol}{value:.2f}", 1)
                pdf.ln()
            
            if chunk != chunks[-1]:
                pdf.add_page()
        
        pdf.output(filename)
        st.success(f"Data exported to PDF successfully with currency: {currency_symbol}!")
        st.download_button(
            label="Download PDF",
            data=open(filename, 'rb').read(),
            file_name=filename,
            mime='application/pdf',
        )
    else:
        st.warning("No data available for the selected date range.")

# Function to generate the balance sheet data
def generate_balance_sheet_data(assets, liabilities, selected_months):
    assets_df = pd.DataFrame(assets, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
    liabilities_df = pd.DataFrame(liabilities, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
    
    if assets_df.empty and liabilities_df.empty:
        return None

    # Combine year and month into a single column for merging
    assets_df['YearMonth'] = assets_df['Year'].astype(str) + '-' + assets_df['Month'].astype(str).str.zfill(2)
    liabilities_df['YearMonth'] = liabilities_df['Year'].astype(str) + '-' + liabilities_df['Month'].astype(str).str.zfill(2)
    
    # Define Category column
    assets_df['Category'] = assets_df['Type'] + ' - ' + assets_df['Nickname'].fillna('')
    liabilities_df['Category'] = liabilities_df['Type'] + ' - ' + liabilities_df['Nickname'].fillna('')
    
    # Filter data by selected months
    assets_df = assets_df[assets_df['YearMonth'].isin(selected_months)]
    liabilities_df = liabilities_df[liabilities_df['YearMonth'].isin(selected_months)]
    
    if assets_df.empty and liabilities_df.empty:
        return None

    # Pivot tables
    assets_pivot = assets_df.pivot_table(index=['Category'], columns='YearMonth', values='Amount', aggfunc='sum').reindex(columns=selected_months, fill_value=0).fillna(0)
    liabilities_pivot = liabilities_df.pivot_table(index=['Category'], columns='YearMonth', values='Amount', aggfunc='sum').reindex(columns=selected_months, fill_value=0).fillna(0)
    
    # Calculate totals
    assets_pivot.loc['Total Assets'] = assets_pivot.sum()
    liabilities_pivot.loc['Total Liabilities'] = liabilities_pivot.sum()
    
    # Calculate net worth
    net_worth = assets_pivot.loc['Total Assets'] - liabilities_pivot.loc['Total Liabilities']
    net_worth.name = 'Net Worth'
    
    # Combine data
    balance_sheet = pd.concat([assets_pivot, liabilities_pivot, net_worth.to_frame().T])
    balance_sheet.reset_index(inplace=True)
    balance_sheet.rename(columns={'index': 'Category'}, inplace=True)
    
    # Ensure the Category column contains only string values
    balance_sheet['Category'] = balance_sheet['Category'].astype(str)
    
    return balance_sheet

# Initialize session state for last inputs
if 'last_asset_nicknames' not in st.session_state:
    st.session_state.last_asset_nicknames = {}

if 'last_liability_nicknames' not in st.session_state:
    st.session_state.last_liability_nicknames = {}

# Define currency list
currencies = {
    "USD": "$",
    "CNY": "¥",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "HKD": "HK$",
    "TWD": "NT$",
    "AUD": "A$",
    "CAD": "C$",
    "SEK": "kr",
    "NZD": "NZ$"
}

def main():
        
    st.title("Personal Finance Dashboard")

    menu = ["Start", "Monthly Balance Sheet", "Investment"]
    choice = st.sidebar.selectbox("Menu", menu)
     # Initialize session state for currency if it hasn't been set yet
    if "currency" not in st.session_state:
        st.session_state.currency = "USD"  # Default value
   
    load_goals()

    # Currency selection widget
    st.sidebar.write("### Settings")
    selected_currency = st.sidebar.selectbox("Select Currency", list(currencies.keys()), index=list(currencies.keys()).index(st.session_state.currency), key="currency")
    st.sidebar.write(f"Selected Currency: {currencies[selected_currency]}")

    # Update session state with the selected currency
    if st.session_state.currency != selected_currency:
        st.session_state.currency = selected_currency

    if choice == "Start":
        st.header("Welcome to the Personal Finance Dashboard!")
        st.write("""
        This application helps you manage your personal finances effectively.
        
        **Monthly Balance Sheet** 

        Before you started, pick a **fixed day** like 30th or 1st to do the financial settlement once a month. It will only cost you 5 minutes to update everytime.
        
        - **Add Asset**: Record your balance of assets
        - **Add Liability**: Record your balance of liabilities.
        - **Visualize Data**: Visualize your financial data with charts, export and save balance sheet.
            (**Attention**: Remember to use "Export Database" after you updated your monthly data, you could retrieve them using "import Database")
        - **Saving Goals**: Set saving goals and track them.
        - **View Data**: View or delete your recorded data.
        
        **Investment** (developing...)\n
                
        Use the menu on the left to navigate through the application.
                 
        This tool is generated mostly by ChatGPT-4o. During this beta stage, your report on bugs will be appreciated. All your data will be stored locally if you choose to save them, no one have access but you.
        
        More features will be added in the future. Have fun!
        
                 
        **Release note (Date: 07/14/2024 Beta 1.0)**
        - Use the "Export Database" and "Import Database" to save and retrive your data after and before you add assets and liabilities, otherwise your data will be lost after app updating.
        - Select the available year-month before your export balance sheet.
        - The history saving goal might be lost after app updating, will be fixed in future version.
        
        """)
    elif choice == "Monthly Balance Sheet":
        submenu = ["Add Asset", "Add Liability", "Visualize Data", "Saving Goals","View Data",]
        sub_choice = st.sidebar.selectbox("Monthly Balance Sheet", submenu)

        if sub_choice == "Add Asset":
            st.subheader("Add Asset")
            year = st.number_input("Year", min_value=2020, max_value=2100, value=2024)
            month = st.number_input("Month", min_value=1, max_value=12, value=6)
            
            type_options = ["Checking Account", "Savings Account", "Investment", "CDs", "Other"]
            type = st.selectbox("Type", type_options + ["Custom..."])
            if type == "Custom...":
                type = st.text_input("Custom Type", value="")
            else:
                st.session_state.last_asset_nicknames[type] = st.session_state.last_asset_nicknames.get(type, "")
            
            nickname = st.text_input("Nickname", value=st.session_state.last_asset_nicknames.get(type, ""))
            amount = st.number_input("Amount", min_value=0.0, value=0.0)
            
            if st.button("Add Asset"):
                st.session_state.last_asset_nicknames[type] = nickname
                insert_asset(year, month, type, nickname, amount)
                st.success("Asset added successfully")

        elif sub_choice == "Add Liability":
            st.subheader("Add Liability")
            year = st.number_input("Year", min_value=2020, max_value=2100, value=2024)
            month = st.number_input("Month", min_value=1, max_value=12, value=6)
            
            type_options = ["Credit Card", "Loan", "Mortgage", "Other"]
            type = st.selectbox("Type", type_options + ["Custom..."])
            if type == "Custom...":
                type = st.text_input("Custom Type", value="")
            else:
                st.session_state.last_liability_nicknames[type] = st.session_state.last_liability_nicknames.get(type, "")
            
            nickname = st.text_input("Nickname", value=st.session_state.last_liability_nicknames.get(type, ""))
            amount = st.number_input("Amount", min_value=0.0, value=0.0)
            
            if st.button("Add Liability"):
                st.session_state.last_liability_nicknames[type] = nickname
                insert_liability(year, month, type, nickname, amount)
                st.success("Liability added successfully")

        elif sub_choice == "View Data":
            st.subheader("View Data")
            
            assets = read_assets()
            liabilities = read_liabilities()

            st.write("### Assets")
            for asset in assets:
                cols = st.columns((4, 1))
                cols[0].write(f"Year: {asset[1]}, Month: {asset[2]}, Type: {asset[3]}, Nickname: {asset[4]}, Amount: {asset[5]}")
                if cols[1].button("Delete Asset", key=f"delete_asset_{asset[0]}"):
                    delete_asset(asset[0])
                    st.success(f"Asset deleted successfully")
                    st.experimental_rerun()  # Refresh the page to update the data

            st.write("### Liabilities")
            for liability in liabilities:
                cols = st.columns((4, 1))
                cols[0].write(f"Year: {liability[1]}, Month: {liability[2]}, Type: {liability[3]}, Nickname: {liability[4]}, Amount: {liability[5]}")
                if cols[1].button("Delete Liability", key=f"delete_liability_{liability[0]}"):
                    delete_liability(liability[0])
                    st.success(f"Liability deleted successfully")
                    st.experimental_rerun()  # Refresh the page to update the data
       
        elif sub_choice == "Visualize Data":
            st.subheader("Visualize Data")
            
            # Export database button
            export_database()
            
            # Import database uploader
            uploaded_file = st.file_uploader("Import Database", type=['db'])
            if uploaded_file is not None:
                import_database(uploaded_file)

            assets = read_assets()
            liabilities = read_liabilities()

            asset_options = format_unique_data_for_selection(assets)
            liability_options = format_unique_data_for_selection(liabilities)

            selected_assets = st.multiselect("Select Assets", asset_options)
            selected_liabilities = st.multiselect("Select Liabilities", liability_options)

            selected_asset_data = [asset for asset in assets if (f"{asset[3]} - {asset[4]}" if asset[4] else asset[3]) in selected_assets]
            selected_liability_data = [liability for liability in liabilities if (f"{liability[3]} - {liability[4]}" if liability[4] else liability[3]) in selected_liabilities]

            st.write("### Select Date Range")
            col1, col2 = st.columns(2)
            start_year = col1.number_input("Start Year", min_value=2020, max_value=2100, value=2024)
            start_month = col1.number_input("Start Month", min_value=1, max_value=12, value=1)
            end_year = col2.number_input("End Year", min_value=2020, max_value=2100, value=2024)
            end_month = col2.number_input("End Month", min_value=1, max_value=12, value=12)

            start_date = pd.to_datetime(f"{start_year}-{start_month}-01")
            end_date = pd.to_datetime(f"{end_year}-{end_month}-01")

            filtered_asset_data = [asset for asset in selected_asset_data if start_date <= pd.to_datetime(f"{asset[1]}-{asset[2]}-01") <= end_date]
            filtered_liability_data = [liability for liability in selected_liability_data if start_date <= pd.to_datetime(f"{liability[1]}-{liability[2]}-01") <= end_date]
   
            selected_months = pd.date_range(start=start_date, end=end_date, freq='MS').strftime("%Y-%m").tolist()

            # Aggregate data for plotting
            asset_df = pd.DataFrame(filtered_asset_data, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
            asset_df['Date'] = pd.to_datetime(asset_df[['Year', 'Month']].assign(DAY=1))
            aggregated_asset_data = asset_df.groupby('Date')['Amount'].sum().reset_index()

            liability_df = pd.DataFrame(filtered_liability_data, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
            liability_df['Date'] = pd.to_datetime(liability_df[['Year', 'Month']].assign(DAY=1))
            aggregated_liability_data = liability_df.groupby('Date')['Amount'].sum().reset_index()

            # Prepare data for plotting
            filtered_asset_data_for_plot = [(row['Date'], row['Amount']) for _, row in aggregated_asset_data.iterrows()]
            filtered_liability_data_for_plot = [(row['Date'], row['Amount']) for _, row in aggregated_liability_data.iterrows()]

            st.write("### Balance Trend")

            with st.container():
                #st.write("Assets Over Time")
                plot_combined_chart(filtered_asset_data_for_plot, "Assets Over Time")

            with st.container():
                #st.write("Liabilities Over Time")
                plot_combined_chart(filtered_liability_data_for_plot, "Liabilities Over Time")

            with st.container():
                #st.write("Net Worth Over Time")
                plot_net_worth(filtered_asset_data, filtered_liability_data, "Net Worth Over Time")

            st.write("### Asset and Liability Overview")

            st.write("#### Select Year and Month for Overview")
            col1, col2 = st.columns(2)
            overview_year = col1.number_input("Year", min_value=2020, max_value=2100, value=2024)
            overview_month = col2.number_input("Month", min_value=1, max_value=12, value=6)

            overview_date = pd.to_datetime(f"{overview_year}-{overview_month}-01")
            filtered_overview_asset_data = [asset for asset in selected_asset_data if pd.to_datetime(f"{asset[1]}-{asset[2]}-01") == overview_date]
            filtered_overview_liability_data = [liability for liability in selected_liability_data if pd.to_datetime(f"{liability[1]}-{liability[2]}-01") == overview_date]

            col1, col2 = st.columns(2)

            with col1:
                #st.write("Asset Overview")
                plot_pie_chart(filtered_overview_asset_data, "Asset Overview")

            with col2:
                #st.write("Liability Overview")
                plot_pie_chart(filtered_overview_liability_data, "Liability Overview")
            
            st.write("### Memo (Optional)")
            memo = st.text_input("Add a memo or your name (optional)")

            # Display Export Buttons        
            currency_symbol = currencies[st.session_state.currency]
            st.write("### Export Balance Sheet")
            if st.button("Export to CSV"):
                export_to_csv(filtered_asset_data, filtered_liability_data, selected_months, start_date, end_date, memo, currency_symbol)
            if st.button("Export to PDF"):
                export_to_pdf(filtered_asset_data, filtered_liability_data, selected_months, start_date, end_date, memo, currency_symbol)

        elif sub_choice == "Saving Goals":
            st.header("Saving Goals")

            # Read asset data
            assets = read_assets()
            asset_options = format_unique_data_for_selection(assets)

            st.write("### Add a New Goal")

            # Form to add a new goal
            with st.form(key='goal_form'):
                name = st.text_input("Goal Name")
                target = st.number_input("Target Amount", min_value=0.0, format="%.2f")
                selected_assets = st.multiselect("Select Assets", asset_options)  # Dropdown for selecting assets
                submit_button = st.form_submit_button(label='Add Goal')

                if submit_button:
                    # Validate input fields
                    if not name:
                        st.warning("Please enter a goal name.")
                    elif target <= 0.0:
                        st.warning("Please enter a target amount greater than 0.")
                    elif not selected_assets:
                        st.warning("Please select at least one asset.")
                    elif any(goal['name'] == name for goal in st.session_state.goals):
                        st.warning("A goal with this name already exists. Please choose a different name.")
                    else:
                        # Get the most recent data for each selected asset
                        if assets:
                            asset_df = pd.DataFrame(assets, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
                            asset_df['Date'] = pd.to_datetime(asset_df[['Year', 'Month']].assign(DAY=1))
                            asset_df = asset_df.sort_values(by='Date', ascending=False).drop_duplicates(subset=['Type', 'Nickname'], keep='first')
                            selected_asset_data = [asset for asset in assets if (f"{asset[3]} - {asset[4]}" if asset[4] else asset[3]) in selected_assets]
                            asset_values = dict(zip([f"{asset[3]} - {asset[4]}" if asset[4] else asset[3] for asset in selected_asset_data], [asset[5] for asset in selected_asset_data]))
                            add_goal(name, target, selected_assets)
                            st.success("Goal added successfully!")
                            st.write(f"Added goal: {name}, Target: {target}, Assets: {selected_assets}")

            # Display existing goals
            if assets:
                asset_values = dict(zip([f"{asset[3]} - {asset[4]}" if asset[4] else asset[3] for asset in assets], [asset[5] for asset in assets]))
                display_goals(asset_values)
            else:
                st.write("No assets available.")

            # Get the most recent date in the asset data
            if len(assets) > 0:
                asset_df = pd.DataFrame(assets, columns=['ID', 'Year', 'Month', 'Type', 'Nickname', 'Amount'])
                asset_df['Date'] = pd.to_datetime(asset_df[['Year', 'Month']].assign(DAY=1))
                most_recent_date = asset_df['Date'].max()
                most_recent_year_month = most_recent_date.strftime('%Y-%m')
                st.write(f"Data until: {most_recent_year_month}")


    elif choice == "Investment":
        st.subheader("Investment Section")
        st.write("Easy to monitor your Stocks, ETFs and other investments")
        st.write("Coming Soon...")
        # Placeholder for investment functionality
    



if __name__ =='__main__':
    main()
