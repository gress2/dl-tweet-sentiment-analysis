from itertools import chain, repeat, islice
from nltk.tokenize import TweetTokenizer, casual
from nltk.stem import porter, snowball
from pandas import read_pickle,to_pickle

def pad_infinite(iterable, padding=None):
    return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
    return islice(pad_infinite(iterable, padding), size)

tokenizer = 'nltk-tweet'
stemmer = 'porter'
pickle_name = 'small'
df = read_pickle('data/' + pickle_name +  '.pd')

print('Cleaning dataframe using tokenizer: {0} and stemmer: {1}'.format(tokenizer, stemmer))
tknzr = None
stmmr = None

if tokenizer == 'nltk-tweet':
    tknzr = TweetTokenizer()
if tokenizer == 'nltk-casual':
    tknzr = casual()
if tknzr is None:
    raise ValueError('invalid tokenizer given. must be one of: nltk-tweet, nltk-casual')

if stemmer == 'porter':
    stmmr = porter.PorterStemmer()
if stemmer == 'snowball':
    stmmr = snowball.SnowballStemmer('english')
if stmmr is None:
    stmmr = { stem: lambda token: token }

df['tokens'] = df.apply(lambda row: list(map(stmmr.stem, tknzr.tokenize(row['SentimentText']))), axis=1)

max_len = 0
for index, row in df.iterrows():
    if len(row['tokens']) > max_len:
        max_len = len(row['tokens'])

df['tokens'] = df.apply(lambda row: list(pad(row['tokens'], max_len, '')), axis=1)

df = df.drop(['SentimentText'], axis=1)

df.to_pickle('data/' + pickle_name + '-cleaned.pd')


