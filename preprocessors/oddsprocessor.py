import argparse

import pandas as pd


teams = {
    'Atlanta':      'Atlanta Hawks',
    'Boston':       'Boston Celtics',
    'Brooklyn':     'Brooklyn Nets',
    'Charlotte':    'Charlotte Hornets',
    'Chicago':      'Chicago Bulls',
    'Cleveland':    'Cleveland Cavaliers',
    'Dallas':       'Dallas Mavericks',
    'Denver':       'Denver Nuggets',
    'Detroit':      'Detroit Pistons',
    'GoldenState':  'Golden State Warriors',
    'Houston':      'Houston Rockets',
    'Indiana':      'Indiana Pacers',
    'LAClippers':   'Los Angeles Clippers',
    'LALakers':     'Los Angeles Lakers',
    'Memphis':      'Memphis Grizzlies',
    'Miami':        'Miami Heat',
    'Milwaukee':    'Milwaukee Bucks',
    'Minnesota':    'Minnesota Timberwolves',
    'NewOrleans':   'New Orleans Pelicans',
    'NewYork':      'New York Knicks',
    'OklahomaCity': 'Oklahoma City Thunder',
    'Orlando':      'Orlando Magic',
    'Philadelphia': 'Philadelphia 76ers',
    'Phoenix':      'Phoenix Suns',
    'Portland':     'Portland Trail Blazers',
    'Sacramento':   'Sacramento Kings',
    'SanAntonio':   'San Antonio Spurs',
    'Toronto':      'Toronto Raptors',
    'Utah':         'Utah Jazz',
    'Washington':   'Washington Wizards',
}


class OddsProcessor:


    def __init__(self, verbose: bool=False):
        self.verbose = verbose

    def process(self, year: int):
        """Load odds Excel File from /data/raw/, and stores a processed one in 
        /data/preprocessed
        """
        inpath = f'../data/raw/odds-{year-1}-{year-2000}.xlsx'
        df = pd.read_excel(inpath)
        
        # First we fix the year in the df
        if self.verbose:
            print('\tParsing monthyear dates')
        
        new_year = False
        for i, monthday in df['Date'].iteritems():
            stringified = str(monthday)
            day = int(stringified[:-2])
            month = int(stringified[:-2])
            if month == 1:
                new_year = True
            ts_year = year if new_year else year - 1
            ts = pd.Timestamp(year=ts_year, month=month, day=day)
            df.at[i, 'Date'] = ts

        # Next we fix the team names
        if self.verbose:
            print('Updating team names')
        
        df['Team'] = df['Team'].map(teams)

        cleaned = pd.DataFrame(index=pd.RangeIndex(len(df) // 2), columns=[
            'date', 'home', 'away', 'home_score', 'away_score', 'ou_open',
            'ou_close', 'home_ml', 'away_ml', 'spread_open', 'spread_close'
        ])
        for i in range(len(df) // 2):
            # Select our two teams
            a = df.iloc[2 * i]
            b = df.iloc[2 * i + 1]
            assert a.at['Date'] == b.at['Date']

            home = a if a.at['VH'] == 'H' else b
            away = b if b.at['VH'] == 'V' else a
            assert not home.equals(away)

            # Manually construct `cleaned` ...blgh
            cleaned.at[i, 'date'] = home.at['Date']
            cleaned.at[i, 'home'] = home.at['Team']
            cleaned.at[i, 'away'] = away.at['Team']
            cleaned.at[i, 'home_score'] = int(home.at['Final'])
            cleaned.at[i, 'away_score'] = int(away.at['Final'])
            cleaned.at[i, 'away_ml'] = int(away.at['ML'])
            cleaned.at[i, 'home_ml'] = int(home.at['ML'])
            
            pk = ('pk', 'PK')
            hopen = 0 if home.at['Open'] in pk else float(home.at['Open'])
            aopen = 0 if away.at['Open'] in pk else float(away.at['Open'])
            hclose = 0 if home.at['Close'] in pk else float(home.at['Close'])
            aclose = 0 if away.at['Close'] in pk else float(away.at['Close'])

            # In terms of structure, the odds files are a mess.
            # We have to do some smart guesswork to pull all the odds
            ou_open = max(hopen, aopen)
            ou_close = max(hclose, aclose)

            spread_open = min(hopen, aopen)
            spread_close = min(hclose, aclose)

            cleaned.at[i, 'ou_open'] = ou_open
            cleaned.at[i, 'ou_close'] = ou_close 
            cleaned.at[i, 'spread_open'] = spread_open
            cleaned.at[i, 'spread_close'] = spread_close 
       
        # Can never have too many sanity checks
        assert (cleaned['spread_open'] < cleaned['ou_open']).all()
        assert (cleaned['spread_close'] < cleaned['ou_close']).all()
        assert len(cleaned) * 2 == len(df)
        
        path = f'../data/preprocessed/{year}-odds.csv'
        cleaned.to_csv(path)
        if self.verbose:
            print(f'{path} written')

if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description='Process Excel files from Sports Book Review Online.'
    )

    parser.add_argument('year', 
        help=('Seasons are identified by their ending year.'), type=int)

    parser.add_argument('-v', 
        help='Flag to display verbose output', action='store_true')
    
    args = parser.parse_args()

    OddsProcessor(args.v).process(args.year)

