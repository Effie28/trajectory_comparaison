import numpy as np
import pandas as pd
import argparse
from pathlib import Path
BASE_PATH = Path(__file__).absolute().parents[1] 

def main (file_name) :

    df = pd.read_csv(BASE_PATH / 'trajectories' / f'{file_name}.csv', names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw", "PDOP", "Sats"], delimiter= ',')
    df.drop(['PDOP', 'Sats'], axis=1, inplace=True)
    df.to_csv(BASE_PATH / 'trajectories' / f'{file_name}_modified.csv', index=False, header=False, sep = ' ')

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