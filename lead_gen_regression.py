import pandas as pd

dataset = pd.read_csv('./lead_gen.csv')
print(dataset.head(20))
print('total data : ', dataset.shape)
print('distinct value of source')
print(dataset['source'].unique())
print('distinct value of country')
print(dataset['country'].unique())