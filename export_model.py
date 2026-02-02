import pyodbc
import configparser

# 1. Read the configuration file
config = configparser.ConfigParser()
config.read('config.ini') 

if 'DATABASE' not in config:
    print("ERROR: [DATABASE] section not found in config.ini")
    exit()

db_config = config['DATABASE']

# CRITICAL: Confirm the DRIVER_NAME matches what is installed on your server
# This is the most common point of failure.
DRIVER_NAME = '{ODBC Driver 17 for SQL Server}' 
SERVER = db_config['SERVER']
DATABASE = db_config['DATABASE']
USER = db_config['USER']
PASSWORD = db_config['PASSWORD']

CONNECTION_STRING = (
    f"DRIVER={DRIVER_NAME};"
    f"SERVER={SERVER};"
    f"DATABASE={DATABASE};"
    f"UID={USER};"
    f"PWD={PASSWORD}"
)

print(f"Attempting to connect to: {SERVER}\\{DATABASE}...")

# 3. Attempt the connection
try:
    cnxn = pyodbc.connect(CONNECTION_STRING)
    cursor = cnxn.cursor()
    print("✅ SUCCESS! Database connection established.")
    
    # Optional: Run a simple query to confirm
    cursor.execute("SELECT GETDATE()")
    row = cursor.fetchone()
    print(f"Database time: {row[0]}")
    
    cnxn.close()

except pyodbc.Error as ex:
    # This output is essential for diagnosis!
    sqlstate = ex.args[0]
    print(f"❌ CONNECTION FAILED!")
    print(f"Error Details: {ex}")
    print("\n--- Diagnostic Checkpoint ---")
    print("1. Check if the SQL Server Browser service is running.")
    print("2. Verify that TCP/IP protocol is enabled in SQL Server Configuration Manager.")
    print("3. Ensure the Windows Firewall on the database server is open for SQL traffic (usually port 1433).")