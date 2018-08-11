import pandas as pd
from functions import *

lemmatizer = nltk.stem.WordNetLemmatizer()


# Preprocessing the input text
inputbigrams = []
inputunigrams = []


def textMessages(inputText):
    sentences = sent_tokenize(inputText)
    for x in sentences:
        punctRemoved = nltk.re.sub(r'[^\w\s]', '', x)

        sentencesLower = punctRemoved.lower()

        sentencesTokenized = nltk.word_tokenize(sentencesLower)
        sentencesNonStop = [x for x in sentencesTokenized if x != []]
        LemmatizedWords = []
        for x in sentencesNonStop:
            LemmatizedWords.append(lemmatizer.lemmatize(x))

        unigram = LemmatizedWords
        bigram = list(ngrams(LemmatizedWords, 2))
        inputunigrams.append(unigram)
        inputbigrams.append(bigram)


if __name__ == '__main__':

    print("Reading File....\n")
    file = pd.read_csv("SMSSpamCollection.tsv", sep="\t", header=None)
    file.columns = ["lablel", "body"]
    file.head()

    print("---- File ----\n")
    print(file.head())
    print("\n")
    print("-----------------------\n")
    print("\n")

    print("Tokenizng Sentences..\n")
    file['sentTokenized'] = sentencesTokenizer(file)
    print(file['sentTokenized'].head())
    print("-----------------------\n")

    print("to lowercase\n")
    file['lowerCased'] = toLowercase(file)
    print(file['lowerCased'].head())
    print("-----------------------\n")

    print("Removing stopwords\n")
    file['stopwordsRemoved'] = removeStopwords(file)
    print(file['stopwordsRemoved'].head())
    print("-----------------------\n")

    print("Tokenizing words..\n")
    file['tokenized'] = wordTokenizer(file)
    print(file['tokenized'].head())
    print("-----------------------\n")

    print("Lemmatizing Words..\n")
    file['lemmatized'] = wordLemmatizer(file)
    print(file['lemmatized'].head())
    print("-----------------------\n")

    print("Creating Bigrams\n")
    file['bigrams'] = toBigram(file)
    print(file['bigrams'].head())
    print("-----------------------\n")

    print("Unigrams\n")
    file['unigrams_flattern'] = toFlatListUnigram(file)
    print(file['unigrams_flattern'].head())
    print("-----------------------\n")

    print("Bigrams\n")
    file['bigrams_flattern'] = toFlatListBigram(file)
    print(file.head())
    print("-----------------------\n")

    print("Unigram Corpus\n")
    unigramCorpus = file.groupby('lable').agg({'unigrams_flattern': 'sum'})
    print(unigramCorpus)
    print("-----------------------\n")

    print("Bigram Corpus\n")
    bigramCorpus = file.groupby('lable').agg({'bigrams_flattern': 'sum'})
    print(bigramCorpus)
    print("-----------------------\n")

    print("\n")
    inputText = input("Please enter something: ")
    print("Preprocessing input text......\n")
    textMessages(inputText)
    print("-----------------------\n")

    inputUnigrams = [item for sublist in inputunigrams for item in sublist]
    inputBigrams = [item for sublist in inputbigrams for item in sublist]

    print("Unigrams from your input\n", inputUnigrams)
    print("\n")
    print("Bigrams from your input\n", inputBigrams)
    print("\n")

    # calculate Bigram Probability
    bigramPSpam, bigramPHam = calculateProbability(unigramCorpus, bigramCorpus, inputUnigrams, inputBigrams)

    print("probability for Spam \n", bigramPSpam)
    print("\n")
    print("probability for Ham \n", bigramPHam)
    print("\n")

    if (bigramPSpam > bigramPHam):
        print("Messeage You entered is a Spam!!!")
    else:
        print("Messeage You entered is a Ham :)")
