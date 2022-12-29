#%%
import yfinance as yf 
papel = yf.Ticker('WEGE3.SA')
dados = papel.history(start = '2010-11-30', end = '2022-11-30')
dados.Dividends.plot()

#%%
dados['month'] = dados.index.month
dados['year'] = dados.index.year
dados['year_month'] = dados['year'].astype(str) + "-" + dados['month'].astype(str)


#%%
#Dividendos por ano 
dados.groupby(['year'])['Dividends'].sum()
dados.groupby(['year'])['Dividends'].sum().plot()

#%%
#Dividendos por mês
dados.groupby(['month'])['Dividends'].sum()

#Número de vezes em que foram pagos dividendos por ano.
dadosfiltrados = dados[dados.Dividends != 0]
dadosfiltrados.groupby(['year'])['Dividends'].count()


