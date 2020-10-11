import argparse 
import os 
import urllib.parse
import time

import pandas as pd
import numpy as np
import requests


BASE_URL = 'https://stats.nba.com'

HEADERS = {
    'Host': 'stats.nba.com',
    'Connection': 'keep-alive',
    'Accept': 'application/json, text/plain, */*',
    'x-nba-stats-token': 'true',
    'User-Agent': (
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 '
        '(KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'),
    'x-nba-stats-origin': 'stats',
    'Sec-Fetch-Site': 'same-origin',
    'Sec-Fetch-Mode': 'cors',
    'Sec-Fetch-Dest': 'empty',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9',
}

def _scrape_schedule(year: int, verbose: bool) -> pd.DataFrame:
    """Scrapes historical NBA schedules from nba.com. This function
    is a little sneaky in that it doesn't directly parse the html. Instead,
    it constructs an artifical referral and reads straight from the 
    json API.
    """
    season_string = f'{year - 1}-{year - 2000}'
    endpoint_params = urllib.parse.urlencode({
        'Counter':          1000,
        'DateFrom':         '',
        'DateTo':           '',
        'Direction':        'DESC',
        'LeagueID':         '00',
        'PlayerOrTeam':     'T',
        'Season':           season_string,
        'SeasonType':       'Regular Season',
        'Sorter':           'DATE'
    })
    url = f'{BASE_URL}/stats/leaguegamelog?{endpoint_params}'
    if verbose:
        print(f'\tRequesting {year} schedule from {BASE_URL}')
    
    referer_params = urllib.parse.urlencode({
        'Season':       season_string, 
        'SeasonType':   f'Regular {year - 2000}Season'
    })
    HEADERS['Referer'] = f'{BASE_URL}/teams/boxscores/?{referer_params}'
    resp = requests.get(url, headers=HEADERS)
    results = resp.json()['resultSets'][0]

    schedule = pd.DataFrame(results['rowSet'], columns=results['headers'])
    if verbose:
        n = len(schedule['GAME_ID'].unique())
        print(f"\tSchedule scraped... found {n} unique games")
    return schedule

def _scrape_single_boxscore(gameid: int, year: int) -> pd.DataFrame:
    """Do the boxscore scraping for a single game"""
    season_string = f'{year - 1}-{year - 2000}'
    endpoint_params = urllib.parse.urlencode({
        'EndPeriod':    10,
        'EndRange':     28800,
        'GameID':       gameid,
        'RangeType':    0,
        'Season':       season_string,
        'SeasonType':   'Regular Season',
        'StartPeriod':  1,
        'StartRange':   0
    })
    url = f'{BASE_URL}/stats/boxscoretraditionalv2?{endpoint_params}'
    HEADERS['Referer'] = f'{BASE_URL}/game/{gameid}/'
    
    resp = requests.get(url, headers=HEADERS)
    results = resp.json()['resultSets'][0]

    boxscore = pd.DataFrame(results['rowSet'], columns=results['headers'])
    return boxscore

def _scrape_boxscores(
        gameids: np.ndarray, year: int, verbose: bool) -> pd.DataFrame:
    """Accumulate the scraped boxscores for all the gameids and
    return a single aggregate dataframe.
    """
    if verbose:
        print(f/t'Requesting boxscores from {BASE_URL}')
    boxscores = pd.DataFrame()
    for i, _id in enumerate(gameids, 1):
        score = _scrape_single_boxscore(_id, year)
        boxscores = boxscores.append(score, ignore_index=True)
        if verbose and i % 50 == 0:
            print(f'\t{i} games scraped')
        # I think I got blacklisted, so let's avoid that
        time.sleep(2)
    if verbose:
        print(f'\tScraped {len(boxscores)} boxscores')
    return boxscores

def scrape_and_save(year: int, verbose: bool) -> None:
    """The function first constructs a dataframe containing high-level
    data for every game, such as score, date, gameid, etc. It then 
    iterates through each gameid and scrapes the boxscore. This function
    writes two separate files, one for schedule df and one for boxscore df.
    """
    schedule = _scrape_schedule(year, verbose)
    gameids = schedule['GAME_ID'].unique()
    boxscores = _scrape_boxscores(gameids, year, verbose)

    sched_out = os.path.join('..', 'data', 'raw', f'{year}-nba-schedule.csv')
    box_out = os.path.join('..', 'data', 'raw', f'{year}-nba-boxscores.csv')
    if args.verbose:
        print(f'\tWriting schedule to {sched_out}')
        print(f'\tWriting boxscores to {box_out}')
    schedule.write_csv(sched_out)
    boxscores.write_csv(box_out)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=
        'Scrape historical boxscore and schedule data from nba.stats.com')

    parser.add_argument('year', type=int, choices=range(2015, 2020),
        help='A season is identified by it\'s ending year')

    parser.add_argument('-v', dest='verbose', action='store_true',
        help='Display verbose output')

    args = parser.parse_args()

    scrape_and_save(args.year, args.verbose)
