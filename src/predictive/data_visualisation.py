import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Clean_Dataset.csv', index_col=0)
df.columns
df['destination_city'].unique()[:10]


# graphique des lignes aériaine par prix
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x=df['airline'], y=df['price'], palette='magma')
plt.title("Top Air lines by Price", fontsize=18, fontweight='bold')
plt.xlabel("Air line")
plt.ylabel("Price")
sns.despine()
plt.show()

# graphique des parts des places éconnomiques et business
plt.figure(figsize=(12, 6))
df1 = df['class'].value_counts()
df1
plt.pie(
    df1,
    labels=['Economy', 'Business'],
    autopct='%.0f%%',
)
plt.show()



# graphique des prix par destination
plt.figure(figsize=(12, 6))
sns.barplot(data=df, x='destination_city', y='price', palette='magma')
plt.show()


## graphique durée par voyage
top = df.groupby(['source_city', 'destination_city'])['duration'].mean().reset_index().sort_values(by='duration').head(
    5)
top
plt.bar(
    top['source_city'] + ' ' + top['destination_city'],
    top['duration'],
    color='skyblue',
)
plt.xticks(rotation=45)
plt.show()