# Pràctica Kaggle APC UAB 2021-22
### Nom: Xavier Val Parejo
### DATASET: NFL-Boxscores
### URL: [kaggle](https://www.kaggle.com/grayengineering425/nfl-box-scores)
## Resum
El dataset utilitza dades dels partits de la NFL compresos entre bastants anys
Tenim X dades amb N atributs. Un % d'ells és categoric / els altres són numérics i estàn normalitzats...
### Objectius del dataset
Volem aprender quina és la correlació que existeix entre les diverses dades que es recullen en un partit de futbol americà i la possiblitat de que l'equip local o visitant guanyi.
## Experiments
Durant aquesta pràctica hem realitzat diferents experiments.
### Preprocessat
Quines proves hem realitzat que tinguin a veure amb el pre-processat? com han afectat als resultats?
### Model
| Model | Hiperparametres | Mètrica | Temps |
| -- | -- | -- | -- |
| [Random Forest](link) | 100 Trees, XX | 57% | 100ms |
| Random Forest | 1000 Trees, XX | 58% | 1000ms |
| SVM | kernel: lineal C:10 | 58% | 200ms |
| -- | -- | -- | -- |
| [model de XXX](link al kaggle) | XXX | 58% | ?ms |
| [model de XXX](link al kaggle) | XXX | 62% | ?ms |
## Demo
Per tal de fer una prova, es pot fer servir amb la següent comanda, si posem 0 no s'executarà aquell model.   
``` python3 demo/demo.py -[svm] 1 -[rf] 0 -[knn] 0 ```

## Conclusions
El millor model que s'ha aconseguit ha estat SVM amb el poly kernel
En comparació amb l'estat de l'art i els altres treballs que hem analitzat....  

## Idees per treballar en un futur
Crec que seria interesant indagar més en l'efecte que pot tenir el temps i en especial el vent en un partit, ho volia aconseguir amb l'altre dataset però no ha estat possible perquè no tenia la correspondència de partits amb el dataset de resultats. També seria interessant intentar ajuntar els diners que es deixen per temporada els equips amb la possiblitat que guanyin un partit.  

## Llicencia
El projecte s’ha desenvolupat sota llicència ZZZz