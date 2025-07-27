import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('merged_cleaned_data.csv')
print(df)

print(df.columns)

df['date'] = pd.to_datetime(df['date'])  

df = df[df['Start Position'] != 0]

df['leverage'] = df['Size USD'] / df['Start Position']

df['is_profitable'] = df['Closed PnL'] > 0

grouped = df.groupby('classification').agg({
    'Closed PnL': ['mean', 'std'],
    'Size USD': 'mean',
    'leverage': 'mean',
    'is_profitable': 'mean'
}).reset_index()

grouped.columns = ['Sentiment', 'Avg_PnL', 'PnL_Risk', 'Avg_SizeUSD', 'Avg_Leverage', 'Profit_Rate']

print(grouped)


sns.barplot(data=grouped, x='Sentiment', y='Profit_Rate')
plt.title("Average Profitability by Market Sentiment")
plt.ylabel("Profit Rate")
plt.show()


sns.barplot(data=grouped, x='Sentiment', y='Avg_Leverage')
plt.title("Average Leverage by Sentiment")
plt.ylabel("Leverage Ratio")
plt.show()

sns.barplot(data=grouped, x='Sentiment', y='PnL_Risk')
plt.title("Risk (PnL Std Dev) by Sentiment")
plt.ylabel("Standard Deviation of PnL")
plt.show()

