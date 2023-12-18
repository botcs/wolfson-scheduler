# Generating Google Cloud Service Account Credentials

Follow these steps to create a service account and obtain its JSON credentials file for accessing Google Sheets through the `gspread` library.

## Google Cloud Console Setup
1. Visit the [Google Cloud Console](https://console.cloud.google.com/).
2. Create a new project or select an existing one.

## Enable Google Sheets API
1. Navigate to the "APIs & Services > Dashboard" section.
2. Click on "+ ENABLE APIS AND SERVICES".
3. Search for "Google Sheets API" and enable it.

## Create Service Account
1. Go to "IAM & Admin > Service Accounts".
2. Click on "Create Service Account".
3. Enter a service account name and description.
4. Click "Create".

## Grant Access to the Service Account (Optional)
- Assign roles if specific permissions are required (optional).
- Optionally, grant users access to this service account.

## Create Keys for Service Account
1. In the service accounts list, find your new account.
2. Go to the "Keys" tab.
3. Click on "Add Key" and select "Create new key".
4. Choose "JSON" as the key type and download the JSON file.

## Configure `service_account.json` in Your Script
- Rename the downloaded JSON file to `service_account.json`.
- Place it in the same directory as your script or update the script's path to the JSON file.

## Share Your Google Sheet (For Google Sheets Access)
- Open your Google Sheet.
- Share it with the email address of the service account (found in the JSON file).

Ensure you handle the JSON file securely as it provides access to your Google Cloud resources.
