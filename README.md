# IA Prédictive

## Sujet du projet 
L'idée du projet est une prédiction de côut d'un vol d'avion

## Présentation du dataset
Le fichier CSV "Clean_Dataset" présente les différents vols d'une destination à une autre selon la compagnie aérienne...  
(Dataset trouvé sur kaggle : https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction)

### Présentation des différentes colonnes
airline : compagnie aérienne
flight : numéro de vol
source_city : ville de départ du vol
departure_time : moment du départ
stops : nombre d'arrêt
arrival_time : moment de l'arrivée
destination_city : ville d'arrivée
class : classe du vol
duration : durée du vol
days_left : nombre de jours avant le départ 
price : prix du billet

### Contenu du dataset
Malheureusement je n'ai trouvé que peu de dataset sur les vol et leur prix en open source, en effet ce sont des données marketing précieuses.  
Il y a donc dans le dataset les villes :
- Mumbai
- Delhi
- Banglagore
- Calcutta
- Hyderabad

## Présentation du code 

### Fichier data_visualisation.py
J'ai pour habitude de faire des graphiques de visualisation afin de m'imprégner du dataset.

### Fichier price_prediction.py
Ce code est l'entrainement de l'IA de prédiction.  
J'ai utilisé ces différentes techniques : 
- L'arbre de décision
- Le random forest
- XGBoost
- LightGBM
- Régression linéaire

### Bibliothèques utilisées  
Pandas : Pour l'importation du fichier CSV  
Matplotlib : Pour faire les graphiques (visualisation dans le fichier data_visualisation.py et visualisation de l'arbre de décision).  
Seaborn : Permet de faire certain graphique tel que les barplot (graphique en baton).  
Scikit-learn : Permet d'entrainé l'IA 

Commandes : 

``pip install pandas``  
``pip install matplotlib``  
``pip install seaborn``  
``pip install scikit-learn``
