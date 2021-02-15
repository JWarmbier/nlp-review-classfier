import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from scipy.sparse import csr_matrix
from collections import Counter
import string
from utils import *

stopwords = ["a", "about", "after", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been",
             "before", "being", "between", "both", "by", "could", "did", "do", "does", "doing", "during", "each",
             "for", "from", "further", "had", "has", "have", "having", "he", "her", "here", "hers", "herself",
             "him",
             "himself", "his", "how", "i", "in", "into", "is", "it", "its", "itself", "let", "me", "more", "most",
             "my",
             "myself", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "own", "sha",
             "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
             "then", "there", "there's", "these", "they", "this", "those", "through", "to", "until", "up", "very",
             "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom", "with", "would", "you",
             "your", "yours", "yourself", "yourselves",
             "n't", "'s", "'ll", "'re", "'d", "'m", "'ve",
             "above", "again", "against", "below", "but", "cannot", "down", "few", "if", "no", "nor", "not", "off",
             "out", "over", "same", "too", "under", "why"]

def extractScoreAndText(reviews):
    reviews = reviews[["score", "reviewText"]]
    reviews = reviews.dropna()
    scores = reviews["score"]
    texts = reviews["reviewText"]
    return scores, texts

class BeforeTokenizationNormalizer():
    @staticmethod
    def normalize(text):
        text = text.strip().lower()
        text = text.replace('&nbsp;', ' ')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('&pound;', u'£')
        text = text.replace('&euro;', u'€')
        text = text.replace('&copy;', u'©')
        text = text.replace('&reg;', u'®')
        return text


def checkIfSpecialToken(word):
    match = re.match(RE_EMOTICONS, word)
    if match is not None:
        return match

    match = re.match(RE_HASHTAG, word)
    if match is not None:
        return match

    match = re.match(RE_HTTP, word)
    return match

class Tokenizer():
    @staticmethod
    def tokenize(text):
        pass

class NltkTokenizer(Tokenizer):
    @staticmethod
    def tokenize(text):
        return word_tokenize(text)

class SimpleTokenizer(Tokenizer):
    @staticmethod
    def tokenize(text):
        wordsInText = re.split(RE_SPACES,text)
        return wordsInText

class ReviewTokenizer(Tokenizer):
    @staticmethod
    def tokenize(text):
        tokens = SimpleTokenizer.tokenize(text)
        i = 0
        while i < len(tokens):
            token = tokens[i]
            match = checkIfSpecialToken(token)

            if match is not None:
                del tokens[i]
                restToken = token.replace(match.string, "")
                if restToken != "":
                    tokens[i:i] = [match.string, restToken]
                else:
                    tokens[i:i] = [match.string]
            else:
                del tokens[i]
                tokens[i:i] = NltkTokenizer.tokenize(token)
            i += 1

        porter = nltk.PorterStemmer()
        tokens = [porter.stem(token) for token in tokens]

        return tokens

def create_bow(reviews, features):
    row = []
    col = []
    data = []

    labels = []

    for i in range(reviews.shape[0]):
        review = BeforeTokenizationNormalizer.normalize(reviews["reviewText"].iat[i])
        score_label = reviews["score"].iat[i]
        review_tokens = ReviewTokenizer.tokenize(review)

        labels.append(score_label)
        for token in set(review_tokens):
            if token not in features:
                continue
            row.append(i)
            col.append(features[token])
            data.append(1)
    return csr_matrix((data, (row, col)), shape=(len(reviews), len(features))), labels

def getUniqueWords(reviews, min_word_count):
    words = Counter()
    for i in range(reviews.shape[0]):
        review = BeforeTokenizationNormalizer.normalize(reviews.iat[i])
        words.update(ReviewTokenizer.tokenize(review))

    words = removeStopwords(words)
    words = removePunctuationMarks(words)
    common_words = list([k for k, v in words.most_common() if v > min_word_count])

    return common_words

def removePunctuationMarks(words):
    punctuationMarks = [punctuation for punctuation in string.punctuation] + ["``", "...", "''"]

    for punctuation in punctuationMarks:
        del words[punctuation]

    return words

def removeStopwords(words):
    for word in stopwords:
        del words[word]

    return words

def prepareCommonWordsDictionary(common_words):
    feature_dict = {}
    for word in common_words:
        feature_dict[word] = len(feature_dict)
    return feature_dict