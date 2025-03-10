from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import sys
import fire
import json
def get_sheet_data(spreadsheet_id, api_key=None, range_name='Sheet1'):
    """
    Reads data from a Google Spreadsheet using an API key.
    
    Args:
        spreadsheet_id (str): The ID of the spreadsheet (can be found in the URL)
        api_key (str): Your Google Sheets API key
        range_name (str): The sheet name or range to read (default: 'Sheet1')
        
    Returns:
        list: A list of dictionaries where keys are column headers and values are cell contents
    """
    if not api_key:
        sys.exit("Please provide an API key")

    try:
        # Build the service with API key authentication
        service = build('sheets', 'v4', developerKey=api_key)
        sheet = service.spreadsheets()
        
        # Call the Sheets API
        result = sheet.values().get(
            spreadsheetId=spreadsheet_id,
            range=range_name
        ).execute()
        values = result.get('values', [])

        if not values:
            print("No data found in spreadsheet.")
            return []

        # Convert to list of dictionaries
        headers = values[0]
        data = []
        
        for row in values[1:]:
            # Pad row with empty strings if it's shorter than headers
            row_data = row + [''] * (len(headers) - len(row))
            row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
            data.append(row_dict)

        return data

    except Exception as e:
        print(f"Error accessing spreadsheet: {e}")
        print("Please verify that:")
        print("1. Your API key is valid")
        print("2. The spreadsheet ID is correct")
        print("3. The spreadsheet is publicly accessible or shared appropriately")
        print("4. steps to get API key:")
        print(f"""
            a. Go to Google Cloud Console (https://console.cloud.google.com)
            b. Create a new project or select existing one
            c. Enable APIs & Services > Library
            d. Search for "Google Sheets API" and enable it
            e. Go to APIs & Services > Credentials
            f. Click "Create Credentials" > "API Key"
            g. Copy your API key
            h. (Optional) Restrict the API key to only Google Sheets API
              """)
        return []

def read_spreadsheet(spreadsheet_url, api_key,range_name='Sheet1'):
    """
    Extracts spreadsheet ID from URL and reads the data.
    
    Args:
        spreadsheet_url (str): The full URL of the Google Spreadsheet
        api_key (str): Your Google Sheets API key
        
    Returns:
        list: A list of dictionaries containing the spreadsheet data
    """
    try:
        # Extract spreadsheet ID from URL
        if '/d/' in spreadsheet_url:
            spreadsheet_id = spreadsheet_url.split('/d/')[1].split('/')[0]
        else:
            raise ValueError("Invalid Google Sheets URL format")
        
        rst=get_sheet_data(spreadsheet_id, api_key,range_name)
    except Exception as e:
        print(f"Error processing URL: {e}")
        rst=[]
        
    return rst

def gen_sheet_to_jsonl(spreadsheet_url, api_key,range_name='Sheet1',output_file='sheet.jsonl'):
    """
    Extracts spreadsheet ID from URL and reads the data.
    
    Args:
        spreadsheet_url (str): The full URL of the Google Spreadsheet
        api_key (str): Your Google Sheets API key
        
    Returns:
        list: A list of dictionaries containing the spreadsheet data
    """
    rst=read_spreadsheet(spreadsheet_url, api_key,range_name)
    with open(output_file,'w') as f:
        for row in rst:
            f.write(json.dumps(row)+'\n')
    print(f"write to {output_file} done")

if __name__ == '__main__':
    fire.Fire(gen_sheet_to_jsonl)