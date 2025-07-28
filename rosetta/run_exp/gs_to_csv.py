from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
import sys
import fire
import json
from rosetta.run_exp.gs_to_jsonl import read_spreadsheet


def gen_sheet_to_csv(spreadsheet_url, api_key,range_name='Sheet1',output_file='sheet.csv'):
    """
    Extracts spreadsheet ID from URL and reads the data.
    
    Args:
        spreadsheet_url (str): The full URL of the Google Spreadsheet
        api_key (str): Your Google Sheets API key
    """
    rst=read_spreadsheet(spreadsheet_url, api_key,range_name)
    # convert a list of dictionaries to a csv file
    keys = rst[0].keys()
    import csv
    with open(output_file, 'w') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(rst)
    

if __name__ == '__main__':
    fire.Fire(gen_sheet_to_csv)