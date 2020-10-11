# bookmaker
A deep learning model for predicting NBA outcomes as a way to gain a competitive
advantage over bookies. The project aims to incorporate novel player-level 
features in addition to traditional team-level statistics to generate
predictions. 

## Data
Pre-processing the data for this project is a non-trivial task. From the start,
we have multiple sources of information. Historical betting odds can be downloaded
from SportsBookReviewOnline. We can easily pull FiveThirtyEight's 
RAPTOR data from their GitHub, but this is all seasonal information so we need more 
granular intra-season data. The `preprocessors` directory contains the Python scripts
we use to retrieve our data the web. The file `nbascraper.py` is the main
scraping script that pulls season schedule and boxscore information by year, and
writes it into `/data/raw/`.

RAPTOR data became available during the 2013-2014 season, and before the 2014
season the Charlotte Bobcats became the Charlotte Hornets. To avoid overly 
complicating the task, we restrict our analysis to the seasons from 2014-2015 
until 2019-2020. As a matter of convention, we typically refer to a season 
by the year it *ended*.
