from ingest_data import read_data
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

pd.options.display.max_columns = 31
games, weather = read_data()

games_dim = games.shape
weather_dim = weather.shape

print(games_dim,weather_dim)

##check what var affects more the result: wind, humidity or temperature
##for date in weather['date']:
  ##  print(date)

## we search for ties and results
## Home win = 1, visitor win = 2, tie = 3
conditions = [games['visitor_score'] < games['home_score'], games['visitor_score'] > games['home_score'], games['visitor_score'] == games['home_score']]
choices = [1,2,3]
games['result'] = np.select(conditions, choices, default=np.nan)

print(games['result'].head(5))

sb.pairplot(games, hue="result")
plt.show()
