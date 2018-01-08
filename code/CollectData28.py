import pandas as pd

row_12 = pd.read_excel('data1,2.xlsx')
row_12.drop(['Ticker', 'Last', 'IVM'], axis=1, inplace=True)
July_21_17 = row_12[1:26].as_matrix()
Sept_15_17 = row_12[27:].as_matrix()

row_34 = pd.read_excel('data3,4.xlsx')
row_34.drop(['Ticker', 'Last', 'IVM'], axis=1, inplace=True)
June_15_18 = row_34[1:26].as_matrix()
Jan_18_19 = row_34[27:].as_matrix()

row_56 = pd.read_excel('data5,6.xlsx')
row_56.drop(['Ticker', 'Last', 'IVM'], axis=1, inplace=True)
May_19_17 = row_56[1:26].as_matrix()
June_16_17 = row_56[27:].as_matrix()

row_78 = pd.read_excel('data7,8.xlsx')
row_78.drop(['Ticker', 'Last', 'IVM'], axis=1, inplace=True)
Oct_20_17 = row_78[1:26].as_matrix()
Jan_19_18 = row_78[27:].as_matrix()





