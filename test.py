from pandas import read_pickle
import clean as Cleaner

df = read_pickle('data/dataset.pd')
Cleaner.clean(df)

