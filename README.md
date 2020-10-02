# bookmaker
A deep learning model for predicting NBA outcomes, and associated betting lines.
This project aims to incorporate novel player-level features in addition to
traditional team-level statistics. 

## Data
Pre-processing the data for this project is a non-trivial task. From the start,
we have multiple sources of information. Historical betting odds can be downloaded
from SportsBookReviewOnline. We can easily pull FiveThirtyEight's 
RAPTOR data from their GitHub, but this is all seasonal information so we need more 
granular intra-season data. The `scrapers` directory contains the Python scripts
we use to retrieve our data from Basketball Reference. First, we pull down
season schedules and their outcomes, and store that in a file. From this, we
have a second scraper that iterates through the games, and scrapes the actual
box scores. We then manually construct 'live' statistics for each player at
every point throughout the season.
