import csv
import json
import sys
import fire

def csv_to_jsonl(input_csv, output_file='output.jsonl'):
    """
    Converts a CSV file to JSONL format.
    
    Args:
        input_csv (str): Path to the input CSV file
        output_file (str): Path to the output JSONL file (default: 'output.jsonl')
        
    Returns:
        int: Number of rows processed
    """
    try:
        # Read the CSV file
        with open(input_csv, 'r', encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            
            # Extract headers from first row
            headers = next(csv_reader)
            
            # Prepare data
            data = []
            for row in csv_reader:
                # Pad row with empty strings if it's shorter than headers
                row_data = row + [''] * (len(headers) - len(row))
                row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
                data.append(row_dict)
        
        # Write to JSONL
        with open(output_file, 'w', encoding='utf-8') as jsonl_file:
            for row in data:
                jsonl_file.write(json.dumps(row) + '\n')
                
        print(f"Successfully converted {len(data)} rows from '{input_csv}' to '{output_file}'")
        return len(data)
        
    except Exception as e:
        print(f"Error converting CSV to JSONL: {e}")
        return 0

if __name__ == '__main__':
    fire.Fire(csv_to_jsonl)