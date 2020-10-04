import argparse
import os

import pandas as pd


class LiveStats:

    def __init__(self, verbose: bool=False):
        self.verbose = verbose

    def run(self, year: int, indir: str, outdir: str):
        """Load the boxscores dataframe and compute cumulative sum
        """
        inpath = os.path.join(indir, f'boxscores{year}.csv')
        boxscores = pd.read_csv(inpath, index_col=0, parse_dates=['date'])

        def _seconds(timestring: str):
            if pd.isnull(timestring):
                return timestring

            minutes, seconds = timestring.split(':')
            return 60 * int(minutes) + int(seconds)

        boxscores['seconds'] = boxscores['mp'].apply(lambda x: _seconds(x))
        boxscores['ngame'] = 1 

        # Drop percentage stats and other derived metrics
        boxscores = boxscores.drop([
            'fg_pct', 'fg3_pct', 'ft_pct', 'br_href',
            'plus_minus', 'mp', 'side', 'starter', 'player'
            ], axis='columns')

        players = boxscores['player_id'].unique()
        season_open = boxscores['date'].min()
        season_close = boxscores['date'].max()

        mi = pd.MultiIndex.from_product([
            pd.date_range(season_open, season_close),
            players
        ])
        
        # Make boxscores have a multiindex
        boxscores = boxscores.set_index(
                ['date', 'player_id'], verify_integrity=True)
        # And then sub in our multiindex, lining up rows
        boxscores = boxscores.reindex(mi).fillna(0)
        
        livestats = boxscores.groupby(level=1).cumsum()
        outpath = os.path.join(outdir, f'{year}-livestats.csv')
        
        if self.verbose:
            print(f'Writing {outpath}')
        livestats.to_csv(outpath)


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=(
            "Read a boxscores .csv file and compute running season totals"
            " for every player for every day in the season")
        )

    parser.add_argument('year', 
        help='Four digit year. A season is defined by its ending year', 
        type=int)

    parser.add_argument('indir',
        help='Relative file path for where box scores are located')

    parser.add_argument('outdir',
        help='Relative file path for where to store generated files')

    parser.add_argument('-v', 
        help='Display verbose output', action='store_true')

    args = parser.parse_args()

    LiveStats(args.v).run(args.year, args.indir, args.outdir)
