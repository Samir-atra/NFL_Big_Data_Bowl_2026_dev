import pandas as pd
import argparse
import os

BASE_DIR = '/home/samer/Desktop/competitions/NFL_Big_Data_Bowl_2026_dev/nfl-big-data-bowl-2026-prediction/train'

def inspect_file(filename):
    # Check if file exists as is, or in base dir
    if os.path.exists(filename):
        filepath = filename
    elif os.path.exists(os.path.join(BASE_DIR, filename)):
        filepath = os.path.join(BASE_DIR, filename)
    else:
        print(f"Error: File '{filename}' not found in current directory or {BASE_DIR}")
        return

    print(f"--- Inspecting: {filepath} ---")
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("\n[Columns]")
    print(df.columns.tolist())

    print("\n[Shape]")
    print(df.shape)

    print("\n[Types]")
    print(df.dtypes)

    print("\n[First 5 Lines]")
    print(df.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect a CSV file.")
    parser.add_argument("filename", help="Name of the file to inspect")
    args = parser.parse_args()

    inspect_file(args.filename)
