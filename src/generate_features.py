from ingest_data import read_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 60
games = read_data()

games_dim = games.shape

def generate_features():

  ## we need to convert a lot of string data to numerical data
  ## we search for ties and results
  ## Home win = 1, visitor win = 2, tie = 3
  conditions = [games['visitor_score'] < games['home_score'], games['visitor_score'] > games['home_score'], games['visitor_score'] == games['home_score']]
  choices = ['1','2','3']
  results = pd.Series(np.select(conditions, choices, default=np.nan).astype('int'))
  #print(games.describe())
  #farem un drop d'aquelles dades amb valors molt baixos que no influiran en el resultat final
  ##visitor_punts_blocked, home_punts_blocked, visitor_penalties, home_penalties
  games.drop(columns=['visitor_punts_blocked', 'home_punts_blocked', 'visitor_penalties', 'home_penalties'])

  ##també hem de preparar altres dades que ens retornen el %
  ##començem amb els sacks, son important perquè determinen una defensa que està dominant
  sacks_away = games['visitor_sack_splits'].str.split('-',n=1,expand=True)
  sacks_home = games['home_sack_splits'].str.split('-',n=1,expand=True)
  sacks_away = pd.Series(sacks_away[1])
  sacks_home = pd.Series(sacks_home[1])
  ##punt_return ens indica si un equip aconsegueix bones posicions de partida
  punt_ret_average_away = games['visitor_punt_return_splits'].str.split('-', n=1, expand=True)
  punt_ret_average_home = games['home_punt_return_splits'].str.split('-', n=1,expand=True)
  punt_ret_average_away = pd.Series(punt_ret_average_away[1])
  punt_ret_average_home = pd.Series(punt_ret_average_home[1])
  ##kick_return fa el mateix que punt_return
  kick_ret_average_away = games['visitor_kick_return_splits'].str.split('-', n=1,expand=True)
  kick_ret_average_home = games['home_kick_return_splits'].str.split('-', n=1,expand=True)
  kick_ret_average_away = pd.Series(kick_ret_average_away[1])
  kick_ret_average_home = pd.Series(kick_ret_average_home[1])
  ##penalties et trenquen el ritme de l'atac i ajuda a la defensa, també ens interessa
  away_yards_penalties = games['visitor_penalty_splits'].str.split('-', n=1, expand=True)
  away_yards_penalties = pd.Series(away_yards_penalties[1])
  home_yards_penalties = games['home_penalty_splits'].str.split('-',n=1,expand=True)
  home_yards_penalties = pd.Series(home_yards_penalties[1])
  #tercer down, pot significar passar de 7 a 3 punts o de perdre l'atac
  third_downs_home = games['home_third_down_splits'].str.split('-',n=2, expand=True)
  third_downs_home[2] = third_downs_home[2].str.rstrip('%').astype('int')
  third_downs_home = pd.Series(third_downs_home[2])
  third_downs_away = games['visitor_third_down_splits'].str.split('-', n=2, expand=True)
  third_downs_away[2] = third_downs_away[2].str.rstrip('%').astype('int')
  third_downs_away = pd.Series(third_downs_away[2])

  #passem els temps de possessio a segons
  home_time_of_possession = "00:" + games['home_time_of_possession']
  home_time_of_possession = pd.to_timedelta(home_time_of_possession).dt.total_seconds()
  away_time_of_possession = "00:" + games['visitor_time_of_possession']
  away_time_of_possession = pd.to_timedelta(away_time_of_possession).dt.total_seconds() 

  #percentatge de FG, interessa més el percentatge que els punts
  home_fg = games['home_field_goals'].str.split('-', n=1, expand=True)
  visitor_fg = games['visitor_field_goals'].str.split('-', n=1,expand=True)
  #hi ha valors nuls representats amb '-', amb la següent linia els tractem
  home_fg = home_fg.replace('', 0)
  home_fg = pd.Series(home_fg[0].astype(float) / home_fg[1].astype(float))
  visitor_fg = visitor_fg.replace('',0)
  visitor_fg = pd.Series(visitor_fg[0].astype(float) / visitor_fg[1].astype(float))

  #intercepcions
  int_yards_away = games['visitor_kick_return_splits'].str.split('-', n=1,expand=True)
  int_home_yards = games['home_kick_return_splits'].str.split('-', n=1,expand=True)
  int_yards_away = pd.Series(int_yards_away[1])
  int_yards_home = pd.Series(int_home_yards[1])

  #faltes, treuen potencia a l'atac
  pen_yards_away = games['visitor_kick_return_splits'].str.split('-', n=1,expand=True)
  pen_yards_home = games['home_kick_return_splits'].str.split('-', n=1,expand=True)
  pen_yards_away = pd.Series(pen_yards_away[1])
  pen_yards_home = pd.Series(pen_yards_home[1])


  header = [
            "results", "Home ToP", "Away ToP", 
            "home Score", "away Score",
            "Home #Plays", "Away #Plays",
            "Home 1st","Away 1st", 
            "Home 1st P", "Home 1st R",
            "Away 1st P", "Away 1st R",
            "Home avg gain", "Away avg gain",
            "Home Net Y" ,"Home Net P", "Home Net R",
            "Away Net Y","Away Net P", "Away Net R", 
            "Home FG", "Away FG",
            "Home 3rd %", "Away 3rd %",
            "Home pens", 
            "Away pens", 
          ]
  test = pd.concat([
            results,home_time_of_possession, away_time_of_possession, 
            games['home_score'], games['visitor_score'],
            games['home_total_plays'], games['visitor_total_plays'],
            games['home_first_downs'],games['visitor_first_downs'],
            games['home_passing_first_downs'], games['home_rushing_first_downs'],
            games['visitor_passing_first_downs'], games['visitor_rushing_first_downs'],
            games['home_avg_gain'], games['visitor_avg_gain'], 
            games['home_net_yards'], games['home_net_yards_passing'], games['home_net_yards_rushing'],
            games['visitor_net_yards'], games['visitor_net_yards_passing'], games['visitor_net_yards_rushing'],
            home_fg, visitor_fg, 
            third_downs_home, third_downs_away,
            games['home_penalties'], 
            games['visitor_penalties'],
            ],
            axis=1, keys=header)


  test.dropna()
  print("Total entrances after dropping null values: ",test.shape[0], ".")
  return test