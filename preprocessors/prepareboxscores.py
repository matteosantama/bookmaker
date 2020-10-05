import argparse
from collections import defaultdict
import os

import pandas as pd


class ParticipantGrouper:

    def run(self, year: int, group: str, verbose: bool=False):
        """This program assumes a `boxscores{year}.csv` exists in the 'raw'
        directory and outputs its data into either 'train' or 'test'.
        """
        boxpath = os.path.join('..', 'data', 'raw', f'boxscores{year}.csv')
        if verbose:
            print(f'Reading boxscores from {boxpath}')
        boxscores = pd.read_csv(boxpath, index_col=0)

        df = boxscores.set_index(['br_href', 'side'])
        
        outpath = os.path.join('..', 'data', group, f'{year}-box.csv')
        if verbose:
            print(f'Writing file to {outpath}')
        df.to_csv(outpath)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
            'Generate files that match a game id to player ids for each year')

    parser.add_argument('year', help='Which year to process. Four digits')
    
    parser.add_argument('group', choices=['train', 'test'])
    
    parser.add_argument('-v', help='Display verbose output', action='store_true')

    args = parser.parse_args()

    ParticipantGrouper().run(args.year, args.group, args.v)
