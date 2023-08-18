import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from datetime import datetime
BASE_PATH = Path(__file__).absolute().parents[1] 

def main (input_file) :

    df = pd.read_csv(BASE_PATH / f'{input_file}.csv', names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"], delimiter= ' ')
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s') # Convert to datetime

    timestamp_initial = df['timestamp'][0]
    df['timestamp'] -= pd.Timedelta(hours=4, seconds=6) # Subtract 4 hours from the RTS Timestamp column

    timestamp_modified = df['timestamp'][0]
    print('\nCorrected timestamp from ' f'{timestamp_initial} to ' f'{timestamp_modified}. Output file : ' f'{input_file}_timestamp_modified.csv')
    df['timestamp'] = df["timestamp"].values.astype(np.int64) / (1e9) # Convert to Unix Timestamp
    
    df.to_csv(BASE_PATH / f'{input_file}_timestamp_modified.csv', index=False, header=False, sep = ' ')

def init_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str, required=True,
                        help='Name of input file.')
    return parser

if __name__ == '__main__':

    parser = init_argparse()
    args = parser.parse_args()
    main(args.input)