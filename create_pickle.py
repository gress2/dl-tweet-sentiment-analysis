import csv
from sys import argv
from pandas import read_csv

if len(argv) is not 3:
    print('Usage: python create_pickle.py <CSV_FILE> <PICKLE_FILE>')
    exit()

csv_file = argv[1] 
pickle_file = argv[2]

df = read_csv(csv_file, delimiter=',', quotechar='"', error_bad_lines=False, 
        doublequote=True, quoting=csv.QUOTE_ALL, skipinitialspace=True, nrows=10000) 
df = df.drop(['SentimentSource'], axis=1)
df.to_pickle(pickle_file)


