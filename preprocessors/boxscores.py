import argparse
from collections import defaultdict
import itertools as it

from bs4 import BeautifulSoup
import pandas as pd
import requests as req


BASE_URL = 'https://www.basketball-reference.com{}'

teams = {
    'Atlanta Hawks':            'ATL',
    'Boston Celtics':           'BOS',
    'Brooklyn Nets':            'BRK',
    'Charlotte Hornets':        'CHO',
    'Chicago Bulls':            'CHI',
    'Cleveland Cavaliers':      'CLE',
    'Dallas Mavericks':         'DAL',
    'Denver Nuggets':           'DEN',
    'Detroit Pistons':          'DET',
    'Golden State Warriors':    'GSW',
    'Houston Rockets':          'HOU',
    'Indiana Pacers':           'IND',
    'Los Angeles Clippers':     'LAC',
    'Los Angeles Lakers':       'LAL',
    'Memphis Grizzlies':        'MEM',
    'Miami Heat':               'MIA',
    'Milwaukee Bucks':          'MIL',
    'Minnesota Timberwolves':   'MIN',
    'New Orleans Pelicans':     'NOP',
    'New York Knicks':          'NYK',
    'Oklahoma City Thunder':    'OKC',
    'Orlando Magic':            'ORL',
    'Philadelphia 76ers':       'PHI',
    'Phoenix Suns':             'PHO',
    'Portland Trail Blazers':   'POR',
    'Sacramento Kings':         'SAC',
    'San Antonio Spurs':        'SAS',
    'Toronto Raptors':          'TOR',
    'Utah Jazz':                'UTA',
    'Washington Wizards':       'WAS'
}

class BoxScoreScraper:
    """BoxScoreScraper processes the output of ScheduleScraper. It expects
    at a minimum, a dataframe indexed by basketball-reference's unique
    box score urls, and containing columns 'home_team_name', 
    'visitor_team_name'.
    """

    def __load_single_boxscore(self, home: str, away: str, href: str):
        """
        Parameters:
            home: home team acronym
            away: away team acronym
            href: basketball-reference's unqiue page id
        """

        gamedata = defaultdict(list)

        url = BASE_URL.format(href)
        res = req.get(url)
        soup = BeautifulSoup(res.content, 'html.parser')

        # Do this for home team and away team
        for acronym, side in zip([home, away], ['HOME', 'AWAY']):
            
            # We store what columns to expect
            cols = []
            colselector = f'#box-{acronym}-game-basic > tbody > tr.thead > th'
            for c in soup.select(colselector):
                cols.append(c['data-stat'])

            # state variable
            is_starter = True

            rowselector = f'#box-{acronym}-game-basic > tbody > tr'
            rows = soup.select(rowselector)
            # Go through each row of the table
            for r in rows:
                player_id = None

                # If we at the seperator, we have gotten to reserves
                if 'class' in r.attrs:
                    is_starter = False
                    continue
                
                # This way we can check if we have run out of data
                for col, tag in it.zip_longest(
                        cols, r.children, fillvalue=None):
                    if tag:
                        # store player id when we encounter it
                        if tag.name == 'th':
                            player_id = tag['data-append-csv']
                        
                        assert tag['data-stat'] in [col, 'reason']
                        if tag['data-stat'] == col:
                            # Force pd.NA instead of None
                            val = tag.string or pd.NA
                            gamedata[col].append(val)
                        else:
                            gamedata[col].append(pd.NA)
                    else:
                        # If no tag, then we add NAs
                        gamedata[col].append(pd.NA)

                assert player_id is not None
                # These stats aren't in the table
                gamedata['player_id'].append(player_id)
                gamedata['side'].append(side)
                gamedata['br_href'].append(href)
                gamedata['starter'].append(is_starter)

        
        df = pd.DataFrame.from_dict(gamedata)
        # Some sanity checks
        assert df['side'].str.contains('HOME').any()
        assert df['side'].str.contains('AWAY').any()
        return df

    def __load_year_scores(
            self, home: pd.Series, away: pd.Series, 
            hrefs: pd.Series, verbose: bool
        ):
        """Compute running player statistics for one NBA season.

        Parameters:
            hrefs: Link information to locate the box score webpage
        """
        df = pd.DataFrame()
        n = len(hrefs)

        # We only need home team, away team, and link to successfully
        # scrape a game
        for i, (h, a, href) in enumerate(zip(home, away, hrefs)):
            home_acronym = teams[h]
            away_acronym = teams[a]

            if verbose:
                print(
                    f'\tReading {a} @ {h}.',
                    f'Game #{i} out of {n} in the season.',
                    sep=' '
                )
            boxscore = self.__load_single_boxscore(
                    home_acronym, away_acronym, href)
            df = df.append(boxscore, ignore_index=True)
        return df

    def run(self, path: str, outdir: str, verbose: bool=False):
        """Scrape box scores for every NBA game and store in a 
        way thats much easier to handle.
        """
        df = pd.read_csv(path, index_col=0)
        seasons = df.groupby('season_year')
        for year, season_schedule in seasons:
            # So now we have all our games grouped by year
            if verbose:
                print(f'Scraping games from {year-1}-{year} season')
            hrefs = season_schedule.index
            home = season_schedule['home_team_name']
            away = season_schedule['visitor_team_name']

            scores = self.__load_year_scores(home, away, hrefs, verbose)

            out_file = f'{outdir}/boxscores{year}.csv'
            scores.to_csv(out_file)


if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('schedule', help=
        'Relative path to schedule file. Should be output of schedules.py')

    parser.add_argument('outdir', help=(
        'This script writes several files, one for each year. This argument '
        'defines the directory that the files should be stored in.'))

    parser.add_argument('-v', help='Flag to print verbose output', 
            action='store_true')

    args = parser.parse_args()

    BoxScoreScraper().run(args.schedule, args.outdir, args.v)
