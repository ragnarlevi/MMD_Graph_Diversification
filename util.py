import numpy as np
import pandas as pd
import datetime


def fetch_raw_data(folder_path):
    """
    Function to fetch all data. Simply to make sure the data is the same in each analysis
    
    """
    price_data = pd.read_csv(folder_path+ "data/raw/YAHOO_PRICE.csv")
    ESG_data = pd.read_excel(folder_path+"data/raw/YAHOO_PRICE_ESG.xlsx", sheet_name= 'ESG')
    asset_profiles = pd.read_excel(folder_path+"data/raw/YAHOO_PRICE_ESG.xlsx", sheet_name= 'asset_profiles') 

    assert np.all(np.isin(np.unique(ESG_data['ticker']),np.unique(price_data['ticker'])))


    esg_series = ESG_data.copy()  
    esg_series['timestamp'] = pd.to_datetime(esg_series['timestamp'])
    min_date = datetime.datetime(2014,10, 1)
    max_date = np.max(ESG_data['timestamp'])
    esg_series['date'] = esg_series['timestamp'].dt.date
    print(min_date)
    print(max_date)
    esg_series


    # Create a series of days, which will then be joined with esg_series, To create "daily" ESG.
    dates = pd.DataFrame({'date':pd.date_range(min_date,max_date,freq='d')})
    dates['date'] = dates['date'].dt.date

    # Pivot the esg data frame to have each row a ESG score at a given date (the index) and the columns a stock frame where index is monthly
    esg_pivot = esg_series[['ticker', 'esgScore', 'date']].copy()
    esg_pivot_diff =pd.pivot_table(esg_pivot, values = 'esgScore', index = 'date', columns= 'ticker').diff().iloc[1:]#  np.log(pd.pivot_table(esg_pivot, values = 'esgScore', index = 'date', columns= 'ticker').pct_change().iloc[1:] + 1)
    esg_pivot = pd.pivot_table(esg_pivot, values = 'esgScore', index = 'date', columns= 'ticker')
    esg_pivot = pd.merge(dates, esg_pivot, how = 'left', left_on='date', right_index=True)
    esg_pivot_diff = pd.merge(dates, esg_pivot_diff, how = 'left', left_on='date', right_index=True)
    esg_pivot.set_index('date', inplace= True)
    esg_pivot_diff.set_index('date', inplace= True)
    esg_pivot = esg_pivot.loc[:, esg_pivot.count() >20]  # remove observations with lass than 20 observations
    esg_pivot_diff = esg_pivot_diff.loc[:, esg_pivot_diff.count() >20]



    # Create a dictionary which keeps track of which companies belong to which sector
    sector_classification = dict()
    for company in asset_profiles['ticker']:
        sector_of_company = asset_profiles['sector'].loc[asset_profiles['ticker'] == company].iloc[0]
        if sector_of_company not in sector_classification.keys():
            sector_classification[sector_of_company] = list()
        sector_classification[sector_of_company].append(company)


    # LOG RETURN
    price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
    price_filtered = price_data.loc[price_data['timestamp'] > datetime.datetime(2014,9, 1), price_data.columns]
    price_filtered['date'] = price_filtered['timestamp'].dt.date

    price_filtered['return'] = price_filtered.sort_values('date').groupby(['ticker']).adjclose.pct_change()

    # set stock as pivot
    adjclose_pivot = price_filtered[['ticker', 'adjclose', 'date']].copy() 
    adjclose_pivot = adjclose_pivot.dropna()  # Drop rows which contain missing values
    adjclose_pivot = pd.pivot_table(adjclose_pivot, values = 'adjclose', index = 'date', columns= 'ticker')

    # Get log return
    price_pivot = price_filtered[['ticker', 'return', 'date']].copy()
    price_pivot['return'] = np.log(1 + price_pivot['return'])
    price_pivot = price_pivot.dropna()  # Drop rows which contain missing values
    price_pivot = pd.pivot_table(price_pivot, values = 'return', index = 'date', columns= 'ticker')

    # Let's order the columns by sector
    company_df = pd.DataFrame({'ticker': np.array(price_pivot.columns)})
    company_df = pd.merge(company_df, asset_profiles, on='ticker', how = 'left')
    company_df['track_index'] = np.array(range(company_df.shape[0]))
    company_df = company_df.sort_values(by = 'sector')
    price_pivot = price_pivot.iloc[:, company_df['track_index']]


    sector_classification = dict()
    for company in asset_profiles['ticker']:
        sector_of_company = asset_profiles['sector'].loc[asset_profiles['ticker'] == company].iloc[0]
        if sector_of_company not in sector_classification.keys():
            sector_classification[sector_of_company] = list()
        sector_classification[sector_of_company].append(company)


    price_pivot = price_pivot.dropna(axis = 1)



    return price_pivot, esg_pivot, sector_classification

