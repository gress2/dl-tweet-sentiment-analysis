'''
For use on english text

options:
    stemmer: 'porter', 'snowball', None
    tokenizer: 'nltk-tweet', 'nltk-casual', None
'''
from nltk.tokenize import TweetTokenizer, casual
from nltk.stem import porter, snowball

def clean(df, tokenizer='nltk-tweet', stemmer='porter'):
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

    df['tokens'] = df.apply(lambda row: map(stmmr.stem, tknzr.tokenize(row['SentimentText'])), axis=1)
    return df.drop(['SentimentText'], axis=1)

