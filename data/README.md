# Data Files
Our original, unprocessed data is stored in the `raw` directory. Those files
include the Excel downloads from Sports Book Review Online, team schedules
scraped from NBA.com, and boxscores for each of those games also
scraped from NBA.com (using `nbascraper.py`). The `raptor.csv` file is
pulled directly from FiveThirtyEight's GitHub, and stores all the relevant 
RAPTOR statistics.

## Train / Test Files
The raw data required a lot of cleaning and processing before it became usable.
All the files within the `train` and `test` directory are prefixed by their
year, which makes retrieving data easier. Generally, we've used NBA.com's IDs 
to identify players and games. Games are identified by their
`GAME_ID`, and players are identified by a `PLAYER_ID`. At some point, we will
have to match NBA.com ID's to BasketballReference ID's because that's what
RAPTOR uses to identify a player. 

## Training Overview
Although much of this will be vectorized in our implementation,
the main logic is:
1. For every year in our training interval...
2. For every game in that season
3. Find the starting players of that game
4. Load current season statistics for those players
5. Load previous season RAPTOR scores from raptor.csv
6. Deep learning magic :) 

Our training years are the 2015 - 2019 seasons, and the 2020 season is our test
year. This is an 80% / 20% split, which despite weighting the training rather
heavily, is still in line with the general wisdom.
