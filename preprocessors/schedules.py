import argparse
from collections import defaultdict
import requests as req

from bs4 import BeautifulSoup
import pandas as pd



BASE_URL ='https://www.basketball-reference.com/leagues/NBA_{}_games-{}.html' 

dtypes = {
    'visitor_team_name':    pd.StringDtype(),
    'visitor_pts':          'int32',
    'home_team_name':       pd.StringDtype(),
    'home_pts':             'int32',
    'box_score_text':       pd.StringDtype(),
    'overtimes':            pd.StringDtype(),
    'attendance':           pd.StringDtype(),   # so we can remove ',' 
    'game_remarks':         pd.StringDtype(),
    'game_start_time':      pd.StringDtype(),   # we can append 'm'
    'br_href':              pd.StringDtype()
}

class ScheduleScraper:
    """
    The ScheduleScraper crawls basketball-reference.com and scrapes historical 
    season schedules and simple box score data. The written dataframe is
    indexed by basketball-reference's game link href, which acts as a
    unique id.
    """

    def __init__(self):
        pass

    def scrape(self, start_year: int, end_year: int, verbose: bool=False):
        """Starting point of all scraping logic. High-level function that 
        delegates tasks such as requesting the webpage, reading the html,
        and parsing the table into a dataframe.

        Returns:
            pd.DataFrame
        """
        df = pd.DataFrame()
        for y in range(start_year, end_year + 1):
            if verbose:
                print(f'Looking up {y-1}-{y} season')
            sched = self.__season_schedule(y, verbose)
            df = df.append(sched)
        return df

    def __season_schedule(self, year: int, verbose=False):
        """Retrieve team schedule and box score for a given season.

        Returns:
            pd.DataFrame
        """
        data = defaultdict(list)
        months = [
            'october', 'november', 'december', 'january', 
            'february', 'march', 'april', 'may', 'june'
        ]
        is_playoffs = False
        for m in months:
            if verbose:
                print(f'\tScraping schdule from {m.capitalize()}')
            url = BASE_URL.format(year, m)
            res = req.get(url)
       
            soup = BeautifulSoup(res.content, 'html.parser')
            table = soup.find(id='schedule')
            
            for row in table.select('tbody > tr'):
                if row.string == 'Playoffs':
                    is_playoffs = True
                    continue
                headers = list(map(lambda x: x['data-stat'], row))
                vals = list(map(lambda x: x.string, row))
               
                # Select box score link
                href = row.select('td:nth-child(7) > a')[0]['href']

                for h, v in zip(headers, vals):
                    data[h].append(v)

                # Add other information since thats not in the table
                data['playoffs'].append(is_playoffs)
                data['br_href'].append(href)

        df = pd.DataFrame.from_dict(data)
        df = df.astype(dtypes)
        if verbose:
            print(f'\tFound {len(df)} total games\n')
        
        # Some additional manipulating we need to do
        att = df['attendance']
        df['attendance'] = att.str.replace(',', '').astype('int32')
        df['date_game'] = pd.to_datetime(
            df['date_game'], format='%a, %b %d, %Y')
        df['game_start_time'] = df['game_start_time'] + 'm'
        df['game_start_time'] = pd.to_datetime(
                df['game_start_time'], format='%I:%M%p').dt.time

        df['season_year'] = year    # So we can easily group games by season
        df = df.drop(['box_score_text', 'game_remarks'], axis='columns')
        df = df.set_index('br_href')
        return df


if __name__=='__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Scrape team schedule data from basketball-reference.com'))

    parser.add_argument('start_year', 
        help=('Four digit starting year. '
            'A season is specified by the year it ended'),
        type=int)

    parser.add_argument('end_year',
            help='Ending year for which to scrape',
            type=int)

    parser.add_argument('output_file', help='Where to store .csv file')
    parser.add_argument('-v', 
            dest='verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()
    scraper = ScheduleScraper()

    df = scraper.scrape(args.start_year, args.end_year, args.verbose)
    if args.verbose:
        print(f'Writing data to {args.output_file}')
    df.to_csv(args.output_file)
