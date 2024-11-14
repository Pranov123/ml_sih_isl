from google.oauth2 import service_account
from googleapiclient.discovery import build

# Replace with the path to your downloaded credentials JSON file
SERVICE_ACCOUNT_FILE = '/content/indiansignlanguage-441714-afc73a9fba0d.json'
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# Folder IDs (replace with your folder IDs)
folder_ids = ['1nkI7lddegUWDvmGsmn68QkB5ikp5_3kt', 'YOUR_SECOND_FOLDER_ID']

# Authenticate and initialize the Drive API
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# Initialize an empty list to store all files
all_files = []

# Function to retrieve files from a specific folder and remove .mp4 extension
def get_files_from_folder(folder_id):
    query = f"'{folder_id}' in parents and mimeType='video/mp4'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get('files', [])
    
    # Remove the .mp4 extension from each file's name
    for file in files:
        file['name'] = file['name'].replace('.mp4', '')
    
    return files

# Retrieve files from each folder and add them to all_files
for folder_id in folder_ids:
    all_files.extend(get_files_from_folder(folder_id))

# Print out each file's name and ID
print(all_files)