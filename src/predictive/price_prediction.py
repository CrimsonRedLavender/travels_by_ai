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

freq = df['destination_city'].value_counts()
df['destination_city'] = df['destination_city'].map(freq)
freq = df['source_city'].value_counts()
df['source_city'] = df['source_city'].map(freq)

X = df.drop(columns=['price'])
y = df['price']


cat_features = [
    'airline', 'source_city', 'destination_city',
    'stops', 'class'
]

num_features = [
    'duration', 'days_left'
]

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', 'passthrough', num_features)
    ]
)


model = LinearRegression()


model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('model', model)
])



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)




y_pred = pipeline.predict(X_test)

print("MAE :", mean_absolute_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))

new_flight = {
    'airline': 'SpiceJet',
    'source_city': 'Delhi',
    'destination_city': 'Mumbai',
    'stops': 'zero',
    'class': 'Economy',
    'duration': 2.17,
    'days_left': 1
}


pipeline.predict(pd.DataFrame([new_flight]))
