# Pràctica Kaggle APC UAB 2021-22
### Nom: Xavier Val Parejo
### DATASET: NFL-Boxscores
### URL: [kaggle](https://www.kaggle.com/grayengineering425/nfl-box-scores)
## Resum
El dataset utilitza dades dels partits de la NFL compresos entre bastants anys.
Tenim *4328* dades amb *57* atributs. Dels 57 atributs trobem 54 numèrics tot i que amb diversos formats i els altres 3 són text que determina la data i els equips que jugaven aquell partit.
### Objectius del dataset
Volem aprender quina és la correlació que existeix entre les diverses dades que es recullen en un partit de futbol americà i la possiblitat de que l'equip local o visitant guanyi.

## Experiments
Durant aquesta pràctica hem realitzat diferents experiments, el primer de tots ha estat agafant tots els atributs més rellevants pel problema però s'ha vist que amb això era massa "fàcil" pel model trovar un 100% d'accuracy i té sentit perquè només mirava els punts dels equips i feia un home_score >= away_score. Després s'ha tret la puntuació de l'equip local per donar una mica més de problemes als models i tot i així han sortit uns resultats interessants.

Si es volen provar els experiments executar la següent comanda, si posem 0 no s'executarà aquell model.  
``` python3 src/test_model.py -[svm] 1 -[rf] 0 -[knn] 0 ```

Ens printarà per consola uns resultats i també farà algunes figures.
Per fer servir el notebook recomano fer correr totes les cel·les de cop.

### Preprocessat
Les proves de preprocessat han estat netejar molt el dataset perquè gran part dels atributs no eren llegibles per la llibreria *pandas* degut al seu format i llavors s'ha passat a un format numèric bàsic per poder fer proves, aquest canvi ha estat per exemple passar el temps de possessió de *mm:ss* a *ssss.ms* o els splits de FG que es pot calcular la seva probabilitat i no tenir un X-X-XX%%.
Han afectat als resultats perquè sense aquests canvis no podriem tenir resultats. 

### Model
| Model | Hiperparametres | Accuracy, F1, Recall | Temps |
| -- | -- | -- | -- |
| Random Forest | 300 Trees, 50 Depth, 0.6 Max_features | 77.86%, 77.71%, 51.46% | 2.89s |
| SVM | kernel:poly C:10 | 72.36%, 72.36%, 47.83%% | 0.35s |
| SVM | kernel:rbf C:10 | 68.96%, 68.74%, 45.32%% | 0.05s |
| KNN | 15 neighbors | 69.24%, 69.15%, 45.73% | 1.16s |

## Demo
Recorda fer ``` pip install -r requirements.txt ```!! En cas que no funcioni prova amb *requirements_old.txt*.
Per tal de fer una prova, es pot fer servir amb la següent comanda, si posem 0 no s'executarà aquell model.   
``` python3 demo/demo.py -[svm] 1 -[rf] 0 -[knn] 0 ```  
En aquest cas se't demanaràn les següents dades, perquè funcioni han de ser totes positives:
- Home ToP: xxxx.x
- Away ToP: xxxx.x
- away Score: xx
- Home 1st R: xx
- Away 1st R: xx
- Home Net R: xxx
- Away Net Y: xxx
- Away Net R: xxx

## Conclusions
El millor model que s'ha aconseguit ha estat SVM amb el poly kernel.
En comparació amb l'estat de l'art i els altres treballs que hem analitzat doncs està bastant bé perquè té uns resultats per sobre del 70% i amb la dificultat que hi ha en els esports per aconseguir la predicció d'un resultat estic bastant content, una cosa que m'ha sorprès ha estat la velocitat però també és d'esperar perquè el dataset no és gaire gran.

## Idees per treballar en un futur
Crec que seria interesant explorar en l'efecte que pot tenir el temps i en especial el vent en un partit, ho volia aconseguir amb l'altre dataset però no ha estat possible perquè no tenia la correspondència de partits amb el dataset de resultats. També seria interessant intentar ajuntar els diners que es deixen per temporada els equips amb la possiblitat que guanyin un partit.  
Un altre idea que m'agradaria haver pogut perseguir és la probabilitat de guanyar segons els equips que estàn jugant el partit.

## Llicencia
El projecte s’ha desenvolupat sota llicència [GPL-3.0 License](https://github.com/xavikp/kaggle-apc2122/blob/main/LICENSE).