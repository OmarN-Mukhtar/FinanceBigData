import requests
from bs4 import BeautifulSoup
import os

url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'

headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/111.0.0.0 Safari/537.36'}

response = requests.get(url, headers = headers)
soup = BeautifulSoup(response.text, 'html.parser')

table = soup.find('table', {'class': 'wikitable'})

tickers = []
for row in table.find_all('tr')[1:]:
  ticker = row.find_all('td')[0].text.replace('\n','').replace('.','-')
  tickers.append(ticker)

out_path = os.path.join(os.path.dirname(__file__), 'sp500_tickers.txt')
with open(out_path, 'w', encoding='utf-8') as f:
    f.write('\n'.join(tickers))
