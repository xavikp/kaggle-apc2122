import pandas as pd
import numpy as np
import os
from pathlib import Path

path = Path(__file__).resolve().parent.parent
data_folder = str(Path(__file__).resolve().parent.parent) + '\data\\'
print (data_folder)

def read_data():
    games = pd.read_csv(data_folder + 'box_scores.csv')
    weather = pd.read_csv(data_folder + 'nfl_game_weather.csv')
    return games, weather

def return_data_info():
    games, weater = read_data()
    print('Total games: ', games.shape[0],'.')
    print('Number of columns from games: ', games.shape[1],'.')

