import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.tree import plot_tree








df = pd.read_csv('Clean_Dataset.csv', index_col=0)
df.columns
df['destination_city'].unique()[:10]

df = df.drop(['arrival_time', 'departure_time'], axis=1)



# graphique des lignes aériaine par prix
'''plt.figure(figsize=(12, 6))
sns.barplot(data=df, x=df['airline'], y=df['price'], palette='magma')
plt.title("Top Air lines by Price", fontsize=18, fontweight='bold')
plt.xlabel("Air line")
plt.ylabel("Price")
sns.despine()
plt.show()'''

# graphique des parts des places éconnomiques et business
'''plt.figure(figsize=(12, 6))
df1 = df['class'].value_counts()
df1
plt.pie(
    df1,
    labels=['Economy', 'Business'],
    autopct='%.0f%%',
)
plt.show()'''



# graphique des prix par destination
'''plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='destination_city', y='price', palette='magma')
plt.show()'''


## graphique durée par voyage
'''top = df.groupby(['source_city', 'destination_city'])['duration'].mean().reset_index().sort_values(by='duration').head(
    5)
top
plt.bar(
    top['source_city'] + ' ' + top['destination_city'],
    top['duration'],
    color='skyblue',
)
plt.xticks(rotation=45)
plt.show()'''


X = df.drop(columns=['price'])
y = df['price']


#défintion des features
cat_features = [
    'airline',
    'source_city',
    'destination_city',
    'stops',
    'class'
]
num_features = [
    'duration',
    'days_left'
]

#préprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features)
    ]
)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#arbre de decision
tree_model = DecisionTreeRegressor(
    max_depth=10,
    min_samples_leaf=20,
    random_state=42
)

#random forest
rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

#xgboost
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='reg:squarederror',
    random_state=42
)

#LightGBM
lgbm_model = LGBMRegressor(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1
)

#régression linéaire
linreg_model = LinearRegression()


pipelines = {
    "Decision Tree": Pipeline([
        ('preprocess', preprocessor),
        ('model', tree_model)
    ]),
    "Random Forest": Pipeline([
        ('preprocess', preprocessor),
        ('model', rf_model)
    ]),
    "XGBoost": Pipeline([
        ('preprocess', preprocessor),
        ('model', xgb_model)
    ]),
    "LightGBM": Pipeline([
        ('preprocess', preprocessor),
        ('model', lgbm_model)
    ]),
    "LingearReg": Pipeline([
        ('preprocess', preprocessor),
        ('model', linreg_model)
    ])
}

#entrainement et évaluation
#erreur moyenne entre le prix réel et le prix prédit
#coefficient de détermination
for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    print(f"\n{name}")
    print("MAE :", mean_absolute_error(y_test, y_pred))
    print("R²  :", r2_score(y_test, y_pred))


#prédiction d'un nouveau vol
new_flight = pd.DataFrame([{
    'airline': 'SpiceJet',
    'source_city': 'Delhi',
    'destination_city': 'Mumbai',
    'stops': 'zero',
    'class': 'Economy',
    'duration': 2.17,
    'days_left': 1
}])
pipelines["LightGBM"].predict(new_flight)


#visualisation de l'arbre de décision
plt.figure(figsize=(20, 10))
plot_tree(
    tree_model,
    filled=True,
    max_depth=3,
    fontsize=10
)
plt.show()

xgb_model.feature_importances_
lgbm_model.feature_importances_
