import pandas as pd

df = pd.read_csv("/teamspace/studios/this_studio/the-protein-engineering-tournament-2023/in_silico_supervised/input/Alpha-Amylase (In Silico_ Supervised)/train.csv")
print(df.info())          # column types, non-null counts
print(df.describe())      # mean, std, min, max, quartiles for numeric cols
print(df.describe(include='object')) 