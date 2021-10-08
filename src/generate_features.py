from ingest_data import read_data
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

pd.options.display.max_columns = 60
games, weather = read_data()

games_dim = games.shape
weather_dim = weather.shape


##check what var affects more the result: wind, humidity or temperature
##for date in weather['date']:
  ##  print(date)

## we search for ties and results
## Home win = 1, visitor win = 2, tie = 3
conditions = [games['visitor_score'] < games['home_score'], games['visitor_score'] > games['home_score'], games['visitor_score'] == games['home_score']]
choices = ['homeW','awayW','tie']
games['result'] = np.select(conditions, choices, default=np.nan)
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


header = ["Home ToP", "Away ToP", "Home First Downs","Away First Downs","Home third downs % comp","Away third downs % comp", "result"]
test = pd.concat([home_time_of_possession, away_time_of_possession, games['home_first_downs'], games['visitor_first_downs'], third_downs_home, third_downs_away, games['result']], axis=1, keys=header)
print(games.head(5))

#datag = pd.concat(data,axis=1,keys=header)

#datag['home_time_of_possession'] = "00:" + datag['home_time_of_possession']
#print(datag['home_time_of_possession'].head(5))
#datag['home_time_of_possession'] = pd.to_timedelta(datag['home_time_of_possession']).dt.total_seconds()
#print(games.head(5))

#sns.regplot(x="home_time_of_possession", y='result', data=datag)
sns.pairplot(test, hue="result")
plt.show()

#sb.pairrplot(games, hue="result")
#plt.show()
