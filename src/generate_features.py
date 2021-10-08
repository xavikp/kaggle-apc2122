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
data = [games['home_time_of_possession'], games['home_score'], games['result']]
header = ["home_time_of_possession","home_score", "result"]
datag = pd.concat(data,axis=1,keys=header)
datag['home_time_of_possession'] = "00:" + datag['home_time_of_possession']
print(datag['home_time_of_possession'].head(5))
datag['home_time_of_possession'] = pd.to_timedelta(datag['home_time_of_possession']).dt.total_seconds()
print(games.head(5))

#sns.regplot(x="home_time_of_possession", y='result', data=datag)
sns.pairplot(datag, hue="result")
plt.show()

#sb.pairrplot(games, hue="result")
#plt.show()
