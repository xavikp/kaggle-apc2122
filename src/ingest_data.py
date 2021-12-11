import pandas as pd
import numpy as np
import os
from pathlib import Path

path = Path(__file__).resolve().parent.parent
data_folder = str(Path(__file__).resolve().parent.parent) + '\data\\'
print (data_folder)

def read_data():
    games = pd.read_csv(data_folder + 'box_scores.csv')
    return games

def return_data_info():
    games = read_data()
    print('Total games: ', games.shape[0],'.')
    print('Number of columns from games: ', games.shape[1],'.')

