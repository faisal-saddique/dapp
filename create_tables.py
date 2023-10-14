import sqlite3
import pandas as pd
import os

# Define the database file name
database_name = "your_database_name.db"

# Connect to the SQLite database
conn = sqlite3.connect(database_name)

# Define table creation queries
create_table_ddw_location = """
CREATE TABLE "DDW_Location_details_with_services_FINAL" (
    "Places_Name" TEXT, 
    "Location Photo" TEXT, 
    "Google Maps link" TEXT, 
    "Places_Latitude" REAL, 
    "Places_Longitude" REAL, 
    "Places_Address" TEXT, 
    "Places_City" TEXT, 
    "Places_PostalCode" TEXT, 
    "Opening times" TEXT, 
    "Services" TEXT, 
    "Dogs allowed" BOOL, 
    "Fully Wheelchair Accessible" BOOL, 
    "Partially Wheelchair Accessible" BOOL, 
    "Toilets available" BOOL, 
    "Wheelchair Friendly Toilet" BOOL, 
    "Wifi available" BOOL
)
"""

create_table_participants = """
CREATE TABLE "Participants_FINAL" (
    "Participant" TEXT, 
    "Programme_Name_Participant_Is_Part_Of" TEXT, 
    "Programme_ID" INTEGER
)
"""

create_table_programme_details = """
CREATE TABLE "Programme_details_with_Narratives_and_Discipline_FINAL" (
    "Programme_ID" INTEGER, 
    "Programme_Name" TEXT, 
    "Programme_Short_Description" TEXT, 
    "Programme_Location" TEXT, 
    "Programme_Access" TEXT, 
    "Programme_Detailed_Description" TEXT, 
    "Image" TEXT, 
    "Page URL" TEXT, 
    "Narratives_Programme_Is_Part_Of" TEXT, 
    "Discipline_Programme_Is_Part_Of" TEXT
)
"""

# Execute the table creation queries
cursor = conn.cursor()
cursor.execute(create_table_ddw_location)
cursor.execute(create_table_participants)
cursor.execute(create_table_programme_details)

# Define the directory containing CSV files
csv_directory = "new data/csvs"

# Define CSV file names
csv_files = [
    "DDW_Location_details_with_services_FINAL.csv",
    "Participants_FINAL.csv",
    "Programme_details_with_Narratives_and_Discipline_FINAL.csv",
]

# Import data from CSV files using pandas
for file in csv_files:
    table_name = os.path.splitext(os.path.basename(file))[0]
    csv_file_path = os.path.join(csv_directory, file)

    df = pd.read_csv(csv_file_path)

    # Save the DataFrame to the SQLite table
    df.to_sql(table_name, conn, if_exists="replace", index=False)

# Commit the changes and close the database connection
conn.commit()
conn.close()

print("Database and tables created, and data imported successfully.")
