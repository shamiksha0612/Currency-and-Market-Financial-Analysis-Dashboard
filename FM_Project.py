import pandas_datareader as pdr
import pandas as pd
from darts import TimeSeries
from darts.models import LinearRegressionModel, Prophet
import matplotlib.pyplot as plt
#%%
# Function to fetch exchange rates and inflation rates
def fetch_data(exchange_ticker, inflation_ticker, interest_ticker, start='2012-01-01'):
    # Fetch data
    exchange_df = pdr.DataReader(exchange_ticker, 'fred', start=start).dropna()
    inflation_df = pdr.DataReader(inflation_ticker, 'fred', start=start).dropna()
    interest_df = pdr.DataReader(interest_ticker, 'fred', start=start).dropna()
    
    # Resample monthly
    exchange_df = exchange_df.resample('M').last()
    inflation_df = inflation_df.resample('M').last()
    interest_df = interest_df.resample('M').last()
    
    # Rename columns for clarity
    exchange_df.rename(columns={exchange_df.columns[0]: 'ExchangeRate'}, inplace=True)
    inflation_df.rename(columns={inflation_df.columns[0]: 'Inflation'}, inplace=True)
    interest_df.rename(columns={interest_df.columns[0]: 'InterestRate'}, inplace=True)
    
    # Merge on date
    data = pd.concat([exchange_df, inflation_df, interest_df], axis=1)
    data = data.dropna()
    return data
#%%
# Function to forecast and plot exchange rates
def forecast_and_plot(exchange_ticker, inflation_ticker, interest_ticker, currency_label, lags=12):
    # Fetch data
    data = fetch_data(exchange_ticker, inflation_ticker, interest_ticker)
    
    # Add lagged features
    for i in range(1, lags + 1):
        data[f'Lag_{i}'] = data['ExchangeRate'].shift(i)
    data = data.dropna()  # Drop NaN rows caused by lagging
    
    # Convert to Darts TimeSeries
    series = TimeSeries.from_dataframe(data, value_cols='ExchangeRate', freq='M')
    inflation_series = TimeSeries.from_dataframe(data, value_cols='Inflation', freq='M')
    interest_series = TimeSeries.from_dataframe(data, value_cols='InterestRate', freq='M')
    features = TimeSeries.from_dataframe(data[[f'Lag_{i}' for i in range(1, lags + 1)] + ['Inflation'] + ['InterestRate']], freq='M')
    
    # Train-test split
    train, val = series.split_before(0.9)
    features_train, features_val = features.split_before(0.9)
    
    # Prophet Model
    prophet_model = Prophet()
    prophet_model.fit(train)
    prophet_forecast = prophet_model.predict(len(val))
    
    # Linear Regression Model
    linear_model = LinearRegressionModel(lags=lags)
    linear_model.fit(series=train)
    linear_forecast = linear_model.predict(len(val))

    
    # Print forecast values for Prophet and Linear Regression
    print(f"\n{currency_label} - Prophet Forecast:")
    print(prophet_forecast.pd_dataframe())
    
    print(f"\n{currency_label} - Linear Regression Forecast:")
    print(linear_forecast.pd_dataframe())
    
    # Plot Predictions
    plt.figure(figsize=(10, 6))
    train.plot(label=f'{currency_label} - Train Data', lw=1)
    val.plot(label=f'{currency_label} - Actual Data', lw=2)
    prophet_forecast.plot(label=f'{currency_label} - Prophet Forecast', lw=2)
    linear_forecast.plot(label=f'{currency_label} - Linear Regression Forecast', lw=2)
    plt.title(f'{currency_label} - Exchange Rate Forecast')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid()
    plt.show()
#%% exchange_ticker, inflation_ticker, interest_ticker
# Forecast for INR
forecast_and_plot('EXINUS', 'INDCPIALLMINMEI', 'INDIRLTLT01STM', 'INR')
# Forecast for CAD
forecast_and_plot('EXCAUS', 'CANCPIALLMINMEI', 'IRLTLT01CAM156N', 'CAD')
# Forecast for UK
forecast_and_plot('EXUSUK', 'GBRCPIALLMINMEI', 'IRLTLT01GBM156N', 'EUR')
# Forecast for JAPAN
forecast_and_plot('EXJPUS', 'JPNCPIALLMINMEI', 'IRLTLT01JPM156N', 'JPY')
# Forecast for swiss
forecast_and_plot('DEXSZUS', 'CHECPALTT01CTGYM', 'IRLTLT01CHM156N', 'CHF')
#%%
# Function to forecast and plot exchange rates for the next 24 months
def forecast_and_plot_future(exchange_ticker, inflation_ticker, interest_ticker, currency_label, lags=12):
    # Fetch data
    data = fetch_data(exchange_ticker, inflation_ticker, interest_ticker)
    
    # Add lagged features
    for i in range(1, lags + 1):
        data[f'Lag_{i}'] = data['ExchangeRate'].shift(i)
    data = data.dropna()  # Drop NaN rows caused by lagging
    
    # Convert to Darts TimeSeries
    series = TimeSeries.from_dataframe(data, value_cols='ExchangeRate', freq='M')
    inflation_series = TimeSeries.from_dataframe(data, value_cols='Inflation', freq='M')
    interest_series = TimeSeries.from_dataframe(data, value_cols='InterestRate', freq='M')
    features = TimeSeries.from_dataframe(data[[f'Lag_{i}' for i in range(1, lags + 1)] + ['Inflation'] + ['InterestRate']], freq='M')
    
    # Train-test split
    train = series  # Use all available data for training
    features_train = features

    # Prophet Model
    prophet_model = Prophet()
    prophet_model.fit(train)
    prophet_forecast = prophet_model.predict(24)  # Predict next 24 months
    
    # Linear Regression Model
    linear_model = LinearRegressionModel(lags=lags)
    linear_model.fit(series=train)
    linear_forecast = linear_model.predict(24)  # Predict next 24 months
    
    # Print forecast values for Prophet and Linear Regression
    print(f"\n{currency_label} - Prophet 24-Month Forecast:")
    print(prophet_forecast.pd_dataframe())
    
    print(f"\n{currency_label} - Linear Regression 24-Month Forecast:")
    print(linear_forecast.pd_dataframe())
    
    # Plot Predictions
    plt.figure(figsize=(10, 6))
    train.plot(label=f'{currency_label} - Historical Data', lw=1)
    prophet_forecast.plot(label=f'{currency_label} - Prophet Forecast', lw=2)
    linear_forecast.plot(label=f'{currency_label} - Linear Regression Forecast', lw=2)
    plt.title(f'{currency_label} - 24-Month Exchange Rate Forecast')
    plt.xlabel('Date')
    plt.ylabel('Exchange Rate')
    plt.legend()
    plt.grid()
    plt.show()
#%%
# Forecast for INR
forecast_and_plot_future('EXINUS', 'INDCPIALLMINMEI', 'INDIRLTLT01STM', 'INR')
# Forecast for CAD
forecast_and_plot_future('EXCAUS', 'CANCPIALLMINMEI', 'IRLTLT01CAM156N', 'CAD')
# Forecast for UK
forecast_and_plot_future('EXUSUK', 'GBRCPIALLMINMEI', 'IRLTLT01GBM156N', 'EUR')
# Forecast for JAPAN
forecast_and_plot_future('EXJPUS', 'JPNCPIALLMINMEI', 'IRLTLT01JPM156N', 'JPY')
# Forecast for Swiss Franc
forecast_and_plot_future('DEXSZUS', 'CHECPALTT01CTGYM', 'IRLTLT01CHM156N', 'CHF')
