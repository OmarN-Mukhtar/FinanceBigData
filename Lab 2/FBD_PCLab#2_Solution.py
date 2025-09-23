# ------------------------------------------------------------------------------------------------------- #
# PCLab 2 Solution: this files focuses on some aspects (not all) of the PC Lab.
# In particular: How to scrap financial data from the web


# If you have any issue understanding my code / if you want a more detailed solution, please contact me: clement.mazetsonilhac@unibocconi.it
# OR, have a look to some of your collegues' solution (on BBoard), they might be better than mine! 
# ------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------- #
# Import packages:
# ------------------------------------------------------------------------------------------------------- #

from xml.dom.expatbuilder import theDOMImplementation
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from copy import copy
from scipy import stats
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go


# ------------------------------------------------------------------------------------------------------- #
# Optional Task: Data-scrapping
# ------------------------------------------------------------------------------------------------------- #

import bs4 as bs
import requests
import yfinance as yf
import datetime

# Use the list of S&P 500 Tickers from wikipedia

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = bs.BeautifulSoup(resp.text, 'lxml')
table = soup.find('table', {'class': 'wikitable sortable'})
tickers = []

for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    tickers.append(ticker)
print(tickers)

# Clean the tickers (remove html stuff)
tickers = [s.replace('\n', '') for s in tickers]

# Prepare the yahoo finance API
start = datetime.datetime(2020, 1, 1)
end = datetime.datetime(2022, 1, 1)
data = yf.download(tickers, start=start, end=end)

# Format the data
df = data.stack().reset_index().rename(index=str, columns={"level_1": "Symbol"}).sort_values(['Symbol','Date'])
df.set_index('Date', inplace=True)

# Alternative way: use Panda datareader
import pandas_datareader as web
df_spy = web.DataReader(tickers,'yahoo',start,end)
df_spy.head()

# ------------------------------------------------------------------------------------------------------- #
# Import data: same data as PCLab#1 !
# ------------------------------------------------------------------------------------------------------- #

stocks_df = pd.read_csv('C:/Users/cms27/Dropbox/FinanceBigData/S1_AssetPricing1/Lab1/Data_PCLab1_Stock.csv')
stocks_df.head()

# Sort the stock data by date
stocks_df = stocks_df.sort_values(by = ['Date'])

# ------------------------------------------------------------------------------------------------------- #
# Task #1 is not covered here.
# ------------------------------------------------------------------------------------------------------- #


# ------------------------------------------------------------------------------------------------------- #
# Other Tasks. Main aspects: computing alpha and beta and testing the CAPM
# ------------------------------------------------------------------------------------------------------- #


# Function to calculate the daily returns 
def daily_return(df):

  df_daily_return = df.copy()
  
  # Loop through each stock
  for i in df.columns[1:]:
    
    # Loop through each row belonging to the stock
    for j in range(1, len(df)):
      
      # Calculate the percentage of change from the previous day
      df_daily_return[i][j] = ((df[i][j]- df[i][j-1])/df[i][j-1]) * 100
    
    # set the value of first row to zero, as previous value is not available
    df_daily_return[i][0] = 0
  return df_daily_return

# Get the daily returns 
stocks_daily_return = daily_return(stocks_df)

# Let's create a placeholder for all betas and alphas (empty dictionaries)
beta = {}
alpha = {}

# Loop on every stock daily return
for i in stocks_daily_return.columns:

  # Ignoring the date and S&P500 Columns 
  if i != 'Date' and i != 'sp500':
    # plot a scatter plot between each individual stock and the S&P500 (Market)
    stocks_daily_return.plot(kind = 'scatter', x = 'sp500', y = i)
    
    # Fit a polynomial between each stock and the S&P500 (Poly with order = 1 is a straight line)
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
    
    plt.plot(stocks_daily_return['sp500'], b * stocks_daily_return['sp500'] + a, '-', color = 'r')
    
    beta[i] = b
    
    alpha[i] = a
    
    plt.show()


# Same but dynamic version:
for i in stocks_daily_return.columns:
  
  if i != 'Date' and i != 'sp500':
    
    # Use plotly express to plot the scatter plot for every stock vs. the S&P500
    fig = px.scatter(stocks_daily_return, x = 'sp500', y = i, title = i)

    # Fit a straight line to the data and obtain beta and alpha
    b, a = np.polyfit(stocks_daily_return['sp500'], stocks_daily_return[i], 1)
    
    # Plot the straight line 
    fig.add_scatter(x = stocks_daily_return['sp500'], y = b*stocks_daily_return['sp500'] + a)
    fig.show()

# Let's view Beta for every stock 
beta
beta_sorted = sorted(beta.items(), key=lambda x: x[1])
beta_sorted 

# Highest beta are AAPL, TESLA, BA, MGM

# Obtain a list of all stock names
keys = list(beta.keys())
keys

# Define the expected return dictionary
ER = {}

rf = 0 # assume risk free rate is zero in this case
rm = stocks_daily_return['sp500'].mean() * 252 # this is the expected return of the market 

for i in keys:
  # Calculate return for every security using CAPM  
  ER[i] = rf + ( beta[i] * (rm-rf) ) 

for i in keys:
  print('Expected Return Based on CAPM for {} is {}%'.format(i, ER[i]))

# The EW portfolio based on the 4 risker assets has a return equal to the EW average of those 4 assets returns

portfolio_weights = np.array([0.25, 0.25, 0, 0.25, 0, 0, 0.25, 0])

# Calculate the portfolio return 
ER_portfolio = sum(list(ER.values()) * portfolio_weights)
ER_portfolio

stocks_daily_return.head()

# Testing the CAPM: create a loop over all year to backtest the CAPM performance
# My code here has been modified and inspired by Group 6 and Group 3 great PC Lab solutions. Check out their code!
 
years = ['2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', '2020']
list_stock = stocks_daily_return.columns.tolist()[1:9]
r_sp500 = stocks_daily_return['sp500']

stock_name = []  
alpha = []
beta = []
realized_return = []
predicted_return1 = []
predicted_return2 = []

for j in list_stock:

    stock = stocks_daily_return[j] # Select data for only current stock.
    
    for i in range(len(years) - 1): 

        curr_year = stock[(stocks_daily_return['Date'] >= (years[i] +'-01-01')) & \
                         (stocks_daily_return['Date'] <= (years[i] +'-12-31'))].reset_index(drop = True)

        curr_year_sp = r_sp500[(stocks_daily_return['Date'] >= (years[i] +'-01-01')) & \
                         (stocks_daily_return['Date'] <= (years[i] +'-12-31'))].reset_index(drop = True)
                         
        out = pd.DataFrame(zip(curr_year, curr_year_sp), columns =['Stock', 'SP500'])

        # Fit a straight line to the data and obtain beta and alpha
        b, a = np.polyfit(out['SP500'], out['Stock'], 1)

        next_year_true = stock[(stocks_daily_return['Date'] >= (years[i+1] +'-01-01')) & \
                         (stocks_daily_return['Date'] <= (years[i+1] +'-12-31'))].reset_index(drop = True)

        next_year_sp = r_sp500[(stocks_daily_return['Date'] >= (years[i+1] +'-01-01')) & \
                         (stocks_daily_return['Date'] <= (years[i+1] +'-12-31'))].reset_index(drop = True)

        # Predict returns using SP500 data of year t (or t-1) and beta of year t-1:
        # Note:no alpha! We stick to the CAPM formula!

        next_year_pred1 = b*(next_year_sp.mean()*252)
        next_year_pred2 = b*(curr_year_sp.mean()*252)

        # Annualize the realized and predicted returns. 
        return_ann_true = next_year_true.mean() * 252


        return_ann_pred1 = next_year_pred1
        return_ann_pred2 = next_year_pred2

        # Store results
        alpha.append(a)
        beta.append(b)
        realized_return.append(return_ann_true)
        predicted_return1.append(return_ann_pred1)
        predicted_return2.append(return_ann_pred2)
        stock_name.append(j)

# Create a dataframe with for every stock, every year, the alpha, beta, predicted return and realized return.

out = pd.DataFrame(zip(stock_name, alpha, beta, realized_return, predicted_return1, predicted_return2), 
                columns = ['Stock', 'Alpha (y-1)', 'Beta (y-1)', 'Realized return (y)', 'Predicted return (y)', 'Predicted return 2 (y)'], \
                   index = years[1:]*8)

# Inspect the output (first 20 entries).


# Construct a scatterplot of the betas and realized returns for all stocks all years.
plt.figure(figsize = (4, 4), dpi = 150)
plt.xlim(0,2)
plt.ylim(-0.55,2)
plt.scatter(out.iloc[:,2], out.iloc[:,3]/100) 
plt.xlabel('Beta (y-1)')
plt.ylabel('Realized return')
betas = np.linspace(0,2,1000)
SML = betas*(stocks_daily_return.sp500.mean()*252/100)
plt.plot(betas, SML, c = "red", zorder = -10)
plt.scatter(1, stocks_daily_return.sp500.mean()*252/100, s = 150, c = "r", zorder = 10)
plt.vlines(x = 1, ymin = stocks_daily_return.mean().min()*252/100 - 0.55, ymax = stocks_daily_return.sp500.mean()*252/100, linestyle = "dashed", color = "gray")
plt.hlines(y = stocks_daily_return.sp500.mean()*252/100, xmin = 0, xmax = 1, linestyle = "dashed", color = "gray")
plt.show()


# Construct a scatterplot of the predicted (v1) and realized returns for all stocks all years.
plt.figure(figsize = (4, 4), dpi = 150)
plt.xlim(-0.5,2)
plt.ylim(-0.5,2)
plt.scatter(out.iloc[:,3]/100,out.iloc[:,5]/100) 
plt.xlabel(' Realized return')
plt.ylabel(' Predicted return')
plt.show()

# Construct a scatterplot of the predicted (v2) and realized returns for all stocks all years.
plt.figure(figsize = (4, 4), dpi = 150)
plt.xlim(-0.5,2)
plt.ylim(-0.5,2)
plt.scatter(out.iloc[:,3]/100,out.iloc[:,4]/100) 
plt.xlabel(' Realized return')
plt.ylabel(' Predicted return')
plt.show()

# Binning is a good way of reducing noise by constructing portfolios of stocks with similar beta:
# Group per 4 observations.
temp = out.sort_values(by = 'Beta (y-1)').reset_index().copy()
temp = temp.groupby(temp.index // 4)[['Beta (y-1)', 'Realized return (y)']].mean()

# Best plot: 

plt.figure(figsize = (4, 4), dpi = 150)
plt.xlim(0,2)
plt.ylim(-0.2,1)
plt.scatter(temp.iloc[:,0], temp.iloc[:,1]/100) 
plt.xlabel('Beta (t-1)')
plt.ylabel('Realized return (t)')
plt.text(1.8, 0.15, "SML", rotation = 0, fontsize = 10, weight = "bold", c = "r")
betas = np.linspace(0,2,1000)
SML = betas*(stocks_daily_return.sp500.mean()*252/100)
plt.plot(betas, SML, c = "red", zorder = -10)
plt.scatter(1, stocks_daily_return.sp500.mean()*252/100, s = 150, c = "r", zorder = 10)
plt.vlines(x = 1, ymin = stocks_daily_return.mean().min()*252/100 - 0.55, ymax = stocks_daily_return.sp500.mean()*252/100, linestyle = "dashed", color = "gray")
plt.hlines(y = stocks_daily_return.sp500.mean()*252/100, xmin = 0, xmax = 1, linestyle = "dashed", color = "gray")
plt.show()
