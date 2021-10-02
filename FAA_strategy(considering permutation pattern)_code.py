import yfinance as yf
import pandas as pd
import numpy as np
import pyfolio as pf
import matplotlib.pyplot as plt

class FAA():
  def __init__(self, level, period, w1, w2, w3, number, transaction_cost, overlap_plot = True, backtest = True):

    self.etf = self.data(level)[0]
    self.etfs = self.data(level)[1]
    self.cash = self.data(level)[2]
    self.momentum = self.ranking(self.factor(self.etf,self.etfs,period)[0],False)
    self.vol = self.ranking(self.factor(self.etf,self.etfs,period)[1],True)
    self.corr = self.ranking(self.factor(self.etf,self.etfs,period)[2],True)
    self.weight_factor = self.weight(self.etf,self.momentum,self.vol,self.corr,w1,w2,w3)
    self.numbers_in_pf = self.etf_num(self.weight_factor,number)
    self.portfolio_change = self.transaction(self.weight_factor,self.numbers_in_pf,self.etf,self.etfs,self.cash,transaction_cost,period, number)

    if overlap_plot == True:
      self.overlap_plot(self.portfolio_change[0],self.portfolio_change[1])
    if backtest == True:
      self.backtest(self.portfolio_change[0],self.portfolio_change[1])

  def data(self,level):

    if level == 'risky':
      etf_name = ['XLY','XLV','XLU','XLP','XLK','XLI','XLF','XLE','XLB','VOX','RWR']
    elif level == 'safe':
      etf_name = ['FXY','FXF','GLD','IEF','SH','TLT','SHY','SHV']

    cash = yf.download('SHV')

    etf =  []
    for i in etf_name:
      etf.append(yf.download(i))

    common_date = []
    for i in range(len(etf)):
      common_date.append(etf[i]['Adj Close'])
    common_date.append(cash['Adj Close'])
    common_date = pd.DataFrame(common_date).T.dropna()

    for i in range(len(etf)):
      etf[i] = etf[i].loc[common_date.index]

    etfs = common_date.iloc[:,:len(etf)]
    cash = common_date.iloc[:,len(etf)]

    return etf, etfs, cash

  def factor(self,etf, etfs, period):
    #permutation_pattern
    pattern = []
    for i in range(len(etf)):
      pattern.append(etf[i].resample('M').last())
      pattern[i]['pattern'] = np.nan
      for j in range(int(len(etf[0].resample('M').last())-period)):
          price = etf[i].loc[etf[i].resample('M').last().index[j]:etf[i].resample('M').last().index[j+period]]['Adj Close']
          price1 = price[1:]
          price2 = price[2:]           
          t = []   
          t1 = [[],[],[],[],[],[]]
          for i1 in range(len(price2)):
              t2 = []
              ranking = []
              ranking.append([price[i1],price1[i1],price2[i1]])
              t2 = [pd.DataFrame(ranking).T.rank()[0].astype(int)[0],pd.DataFrame(ranking).T.rank()[0].astype(int)[1],pd.DataFrame(ranking).T.rank()[0].astype(int)[2]]
              if t2 == [1, 2, 3]:
                  t1[0].append('A')
              elif t2 == [1, 3, 2]:
                  t1[1].append('B')
              elif t2 == [2, 1, 3]:
                  t1[2].append('C')
              elif t2 == [2, 3, 1]:
                  t1[3].append('D')
              elif t2 == [3, 1, 2]:
                  t1[4].append('E')
              elif t2 == [3, 2, 1]:
                  t1[5].append('F')
          t3 = []
          for j1 in range(len(t1)):
              t3.append((len(t1[j1])/len(price2))*np.log((len(t1[j1])+1)/(len(t1[5-j1])+1)))
          t.append(sum(t3))           
          pattern[i]['pattern'][j+period] = t[0]
      pattern[i] = pattern[i]['pattern'].dropna()

    #volatility
    vol = []
    for i in range(len(etf)):
      vol.append(etf[i].resample('M').last())
      vol[i]['vol'] = np.nan
      for j in range(int(len(etf[0].resample('M').last())-period)):
        vol[i]['vol'][j+period] = etf[i].loc[etf[i].resample('M').last().index[j]:etf[i].resample('M').last().index[j+period]]['Adj Close'].std()
      vol[i] = vol[i]['vol'].dropna()

    #correlation
    corr = []
    for i in range(len(etf)):
      corr.append(etf[i].resample('M').last())
      corr[i]['corr'] = np.nan
      for j in range(int(len(etf[0].resample('M').last())-period)):
        corr[i]['corr'][j+period] = etfs.loc[etfs.resample('M').last().index[j]:etfs.resample('M').last().index[j+period]]['Adj Close'].corr().sum()[i]
      corr[i] = corr[i]['corr'].dropna()
    
    return pattern, vol, corr

  def ranking(self,x,TF):
    for i in range(len(x[0])):
      rank_etfs = []
      for j in range(len(x)):
        rank_etfs.append(x[j].iloc[i])
      rank_etfs = pd.DataFrame(rank_etfs).rank(ascending=TF)
      for j in range(len(x)):
        x[j].iloc[i] = rank_etfs.iloc[j][0]
    return x

  def weight(self,etf,x,y,z,w1,w2,w3):
    weight_given_rank = []
    for i in range(len(etf)):
      weight_given_rank.append(x[i]*w1 + y[i]*w2 + z[i]*w3)
    return weight_given_rank

  def etf_num(self,x,number):
    final_rank = []
    for i in range(len(x[0])):
      rank_etfs = []
      for j in range(len(x)):
        rank_etfs.append(x[j].iloc[i])
      rank_etfs = pd.DataFrame(rank_etfs).rank()
      final_rank.append(rank_etfs[rank_etfs<=number].dropna().index)
    return final_rank

  def transaction(self,x,y,etf,etfs,cash,transaction_cost,period,number):
    s1 = []
    s2 = []
    for j in range(len(x[0].index.strftime('%Y-%m')[1:])):
      Absolute_momentum = (etfs.resample('M').last().iloc[:,y[j]].iloc[j+period] - etfs.resample('M').last().iloc[:,y[j]].iloc[j])
      Absolute_momentum.index = y[j] 
      etfs_change = etfs.pct_change().loc[x[0].index.strftime('%Y-%m')[1:][j]].iloc[:,Absolute_momentum[Absolute_momentum>0].index]
      cash_change = cash.pct_change().loc[x[0].index.strftime('%Y-%m')[1:][j]]
      if len(etfs_change.columns) <= number:
        port_change = etfs_change.T.sum()/number + ((number-len(etfs_change.columns))*cash_change).sum()/number
        port_change.iloc[0] = port_change.iloc[0] - transaction_cost
        s1.append(port_change)
      else:
        port_change = etfs.pct_change().loc[x[0].index.strftime('%Y-%m')[1:][j]].iloc[:,y[j]].T.sum()/len(y[j])
        port_change.iloc[0] = port_change.iloc[0] - transaction_cost
        s1.append(port_change)
      equal_weight = (etfs.pct_change().loc[x[0].index.strftime('%Y-%m')[1:][j]].T.sum())/len(etf)
      equal_weight.iloc[0] = equal_weight.iloc[0] - transaction_cost
      s2.append(equal_weight)
    return s1, s2

  def overlap_plot(self, FAA, Equal_weight):
    plt.plot((pd.concat(FAA[0:len(FAA)],axis=0)+1).cumprod(), label = 'FAA')
    plt.plot((pd.concat(Equal_weight[0:len(Equal_weight)],axis=0)+1).cumprod(), label = 'Equal weight')
    plt.legend()
    plt.show()

  def backtest(self, FAA, Equal_weight):
    pf.create_returns_tear_sheet(pd.concat(FAA[0:len(FAA)],axis=0))
    pf.create_returns_tear_sheet(pd.concat(Equal_weight[0:len(Equal_weight)],axis=0))