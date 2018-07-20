# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 21:51:22 2018

@author: Sahil S
"""

import pandas as pd
import numpy as np

 
class MonteCarlo:
    def __init__(self,*files):
        self.stocks = ['s1','s2','s3','s4','s5']
        self.files = files
        self.price_data = []
        self.start_price = []            
        self.daily_mean_DF = pd.DataFrame(index = self.stocks, columns = ['Values'])
        self.daily_stdDev_DF = pd.DataFrame(index = self.stocks, columns = ['Values'])
        self.daily_mean_vector = []
        self.daily_stdDev_vector = []
        self.quantity = []
        self.meanPrice = []
        self.stdDeviation  = []        
        
    def get_price_data_of5stocks(self):
        for i in range(len(self.files)):
            if ( i == 0):
                df_priceData = pd.read_csv(self.files[i])
            elif(i == 1):
                df_quantity = pd.read_csv(self.files[i])
        price_data = df_priceData.iloc[:,1:6]
        self.price_data = price_data
        self.price_data = self.price_data.T 
        start_data = price_data.iloc[-1,:]
        self.start_price = start_data
        quantity = df_quantity.iloc[:,2]
        self.quantity = quantity.iloc[0:5,]
      
    def calc_mean_stdDev_returns(self):
        for i in range(len(self.stocks)):
            returns  = []
            for j in range(len(self.price_data[i]) - 1):
                returns.append((self.price_data[i+1]/self.price_data[i])-1)
            log_returns = np.log(1+np.asarray(returns))
            self.daily_mean_vector.append(np.mean(log_returns))                
            self.daily_stdDev_vector.append(np.std(log_returns))
        self.makeDataframes()
    
    def makeDataframes(self):
        self.daily_mean_DF['Values'] = self.daily_mean_vector
        self.daily_stdDev_DF['Values'] = self.daily_stdDev_vector

    # extracting mu from historical data
    def historical_mu(self, stock):
        return self.daily_mean_DF.loc[stock]
        
    # extracting sigma from historical data    
    def historical_sigma(self, stock):
        return self.daily_stdDev_DF.loc[stock]        
    
    def GBMGenerator(self,stock):
        n = 252
        mu = self.historical_mu(stock)
        mu = np.asarray(mu)
        sigma = self.historical_sigma(stock)
        sigma = np.asarray(sigma)
        S0 = self.start_price.loc[stock]
        T = 1
        dt =  float (T) / n
        
        ## S_T
        X = np.arange(0,1,dt)
        X = np.hstack((X,np.array([1.0])))
        S_T = []
        for i in range(0,100):
            Z = np.random.standard_normal(size = n)
            W_t = np.cumsum(Z)
            W_t = np.hstack((np.array([0.0]), W_t))            
            temp = (mu - ((sigma**2)/2))
            p1 = (temp)*X
            p2 = (sigma) * np.sqrt(dt) * W_t
            S_t = S0 * np.exp(p1+p2)
            S_T.append(S_t[n])
        
        average_stock_price = np.mean(S_T)
        return average_stock_price;
                
    def Monte_carlo(self):
        meanPriceList = []
        for i in range(len(self.stocks)):
#           price = []            
            mean_price = self.GBMGenerator(self.stocks[i])
#           price.append(stock_price)                
#            mean_price = np.mean(price)
            meanPriceList.append(mean_price)
        self.meanPrice = meanPriceList
        
    def calculate_Expected_payoff(self):
        total_basket_price = 0 
        K = 500000
        n = 0
        sum_payoffs = 0
#        while n < 100:
        self.Monte_carlo()            
        total = np.asarray(self.quantity) * np.asarray(self.meanPrice)
        total_basket_price = np.sum(total)
        payoff = max((total_basket_price - K), 0)    
        sum_payoffs += payoff
#            n += 1
        expected_payoff = sum_payoffs
        return expected_payoff     
        
mc = MonteCarlo("priceHistory.csv", "VolsShares.csv")
mc.get_price_data_of5stocks()
mc.calc_mean_stdDev_returns()

expected_payoff = mc.calculate_Expected_payoff()
price = expected_payoff * np.exp(-0.02*1)
print("The expected payoff", expected_payoff)
