import ast
import string
import nltk

from nltk import sent_tokenize
from nltk.corpus import stopwords
from nltk import ngrams
from main import lemmatizer

stop = stopwords.words('english')


# tokenizing sentences and removing punctuation
def sentencesTokenizer(file):
    punctuations = string.punctuation
    punctuations = punctuations.replace(',', '')
    TokenizedSentences = file['body'].apply(sent_tokenize)
    ff = lambda sent: ''.join(ch for w in sent for ch in w
                             if ch not in string.punctuation)

    TokenizedSentences = TokenizedSentences.apply(lambda row: list(map(ff, row)))
    return TokenizedSentences

# to lowercase
def toLowercase(file):
    lc = file['sentTokenized'].astype(str).str.lower().transform(ast.literal_eval)
    return lc

# Remove stopwords
def removeStopwords(file):
    sr = file['lowerCased'].astype(str).apply(
        lambda x: ' '.join([word for word in x.split() if word not in (stop)])).transform(ast.literal_eval)
    return sr

# Tokenizing to words
def wordTokenizer(file):
    wordTokenizer = nltk.word_tokenize
    tt = file['stopwordsRemoved'].apply(lambda row: list(map(wordTokenizer, row)))
    return tt

# Lemmatizing words
def wordLemmatizer(file):
    ll = file['tokenized'].apply(
        lambda row: list(list(map(lemmatizer.lemmatize, y)) for y in row))
    return ll

# Creating bigrams
def toBigram(file):
    bigram = file['lemmatized'].apply(
        lambda row: list(map(lambda x: list(ngrams(x, 2)), row)))
    return bigram

def toFlatListUnigram(file):
    fu = file['lemmatized'].apply(
        lambda row: [item for sublist in row for item in sublist])
    return fu

def toFlatListBigram(file):
    flb = file['bigrams'].apply(
        lambda row: [item for sublist in row for item in sublist])
    return flb

def calculateProbability(unigramCorpus, bigramCorpus, inputUnigrams, inputBigrams):

    unigramCount = unigramCorpus.assign(count=unigramCorpus.unigrams_flattern.apply(lambda x: len(set(x))))
    Spam = unigramCount.at['spam', 'count']
    Ham = unigramCount.at['ham', 'count']

    bigramPSpam = 1
    bigramPHam = 1

    #Calculating probability using Spam corpus
    for x in range(len(inputBigrams)-1):
        bigramPSpam *= (((bigramCorpus.loc["spam", "bigrams_flattern"].count(inputBigrams[x])) + 1) / (
            (unigramCorpus.loc["spam", "unigrams_flattern"].count(inputUnigrams[x]) + Spam)))

    #Calculating probability using Ham forpus
    for x in range(len(inputBigrams)-1):
        bigramPHam *= (((bigramCorpus.loc["ham", "bigrams_flattern"].count(x)) + 1) / (
            (unigramCorpus.loc["ham", "unigrams_flattern"].count(inputUnigrams[x]) + Ham)))

    return (bigramPSpam, bigramPHam)
