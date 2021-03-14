import pandas as pd
import numpy as np

df = pd.read_csv('/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project/data/SA1015.csv')
chromosome = sorted(df['chr'].unique())
chromosome = chromosome[:len(chromosome)-1]
print(chromosome)
print(df['chr'] == 1)

for i in chromosome:
    dat = df[df['chr'] == i]
    dat = pd.DataFrame(dat)
    dat.to_csv('/Users/xiaoxuanliang/Desktop/STAT 520A/STAT-520A-Project/data/chr_%d.csv' %i)

