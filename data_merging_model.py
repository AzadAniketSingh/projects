import pandas as pd

trader_data = pd.read_csv('historical_data.csv')
fear_greed = pd.read_csv('fear_greed_index.csv')

trader_data = trader_data[['Execution Price', 'Size Tokens', 'Size USD', 'Side',
                           'Timestamp IST', 'Start Position', 'Closed PnL', 'Fee', 'Direction']]

fear_greed = fear_greed[['value', 'classification', 'date']]


trader_data['date'] = pd.to_datetime(trader_data['Timestamp IST'], format='%d-%m-%Y %H:%M').dt.date

fear_greed['date'] = pd.to_datetime(fear_greed['date']).dt.date

trader_data.dropna(inplace=True)
fear_greed.dropna(inplace=True)

merged_df = pd.merge(trader_data, fear_greed, on='date', how='inner')


merged_df.to_csv('merged_cleaned_data.csv', index=False)
print("Cleaned data saved to 'merged_cleaned_data.csv'")
