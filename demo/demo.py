import argparse
import sys
from pathlib import Path
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score, recall_score

p = str(Path(__file__).parent.parent)
p = p + '\src'
sys.path.insert(0, p)

import train_model as tm
from generate_features import generate_features
from clean_dataset import prune_attr

parser = argparse.ArgumentParser()
parser.add_argument("-svm", help="Use SVM with poly kernel", default=0)
parser.add_argument("-rf", help="Use Random Forest classifier", default=0)
parser.add_argument("-knn", help="User KNN method", default=0)
args = parser.parse_args()

df = generate_features()
df = prune_attr(df)



'''
Necessary data (All values are positive):
Home ToP: xxxx.x
Away ToP: xxxx.x
away Score: xx
Home 1st R: xx
Away 1st R: xx
Home Net R: xxx
Away Net Y: xxx
Away Net R: xxx
'''
if(args.svm != 0):
    hTop = pd.Series(float(input("Home ToP:")))
    aTop = pd.Series(float(input("Away ToP:")))
    aSco = pd.Series(int(input("Away Score:")))
    hFR = pd.Series(int(input("Home 1st Rushing:")))
    aFR = pd.Series(int(input("Away 1st Rushing:")))
    hNR = pd.Series(int(input("Home Net Rushing Yards:")))
    aNY = pd.Series(int(input("Away Net Yards:")))
    aNR = pd.Series(int(input("Away Net Rushing Yards:")))
    
    header = [
            "Home ToP", "Away ToP", 
            "away Score",
            "Home 1st R",
            "Away 1st R",
            "Home Net R",
            "Away Net Y","Away Net R", 
          ]
    game = pd.concat([
            hTop, aTop, aSco, hFR, aFR, hNR, aNY, aNR
            ],
            axis=1, keys=header)

    poly, rbf, x_test, y_test, y_train = tm.train_SVM(df, False)
    y_pred = poly.predict(game)
    if(y_pred != 1):
        print("SVM with poly kernel predicts the result to be victory to the AWAY team!")
    else:
        print("SVM with poly kernel predicts the result to be victory to the HOME team!")

    y_pred = rbf.predict(game)
    if(y_pred != 1):
        print("SVM with rbf kernel predicts the result to be victory to the AWAY team!")
    else:
        print("SVM with rbf kernel predicts the result to be victory to the HOME team!")
    
    if(rbf.predict(game) != poly.predit(game)):
        print("Seems that there's some discrepancies...")


    

if(args.rf != 0):
    hTop = pd.Series(float(input("Home ToP:")))
    aTop = pd.Series(float(input("Away ToP:")))
    aSco = pd.Series(int(input("Away Score:")))
    hFR = pd.Series(int(input("Home 1st Rushing:")))
    aFR = pd.Series(int(input("Away 1st Rushing:")))
    hNR = pd.Series(int(input("Home Net Rushing Yards:")))
    aNY = pd.Series(int(input("Away Net Yards:")))
    aNR = pd.Series(int(input("Away Net Rushing Yards:")))
    
    header = [
            "Home ToP", "Away ToP", 
            "away Score",
            "Home 1st R",
            "Away 1st R",
            "Home Net R",
            "Away Net Y","Away Net R", 
          ]
    game = pd.concat([
            hTop, aTop, aSco, hFR, aFR, hNR, aNY, aNR
            ],
            axis=1, keys=header)

    random_forest, x_test, y_test, y_train = tm.train_RF(df, 300)
    y_pred = random_forest.predict(game)
    if(y_pred != 1):
        print("Random Forests predicts the result to be victory to the AWAY team!")
    else:
        print("Random Forests predicts the result to be victory to the HOME team!")


if(args.knn !=0):
    hTop = pd.Series(float(input("Home ToP:")))
    aTop = pd.Series(float(input("Away ToP:")))
    aSco = pd.Series(int(input("Away Score:")))
    hFR = pd.Series(int(input("Home 1st Rushing:")))
    aFR = pd.Series(int(input("Away 1st Rushing:")))
    hNR = pd.Series(int(input("Home Net Rushing Yards:")))
    aNY = pd.Series(int(input("Away Net Yards:")))
    aNR = pd.Series(int(input("Away Net Rushing Yards:")))
    
    header = [
            "Home ToP", "Away ToP", 
            "away Score",
            "Home 1st R",
            "Away 1st R",
            "Home Net R",
            "Away Net Y","Away Net R", 
          ]
    game = pd.concat([
            hTop, aTop, aSco, hFR, aFR, hNR, aNY, aNR
            ],
            axis=1, keys=header)

    knn, x_test, y_test, y_train = tm.train_KNN(df, 15)
    y_pred = knn.predict(game)
    if(y_pred != 1):
        print("KNN predicts the result to be victory to the AWAY team!")
    else:
        print("KNN predicts the result to be victory to the HOME team!")