import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pickle
from collections import Counter
matplotlib.rcParams.update({'font.size': 14})

figuresDir = "figures/"

RE_SPACES = re.compile("\s+")
RE_HASHTAG = re.compile("[@#][_a-z0-9]+")
RE_EMOTICONS = re.compile("(:-?\))|(:p)|(:d+)|(:-?\()|(:/)|(;-?\))|(<3)|(=\))|(\)-?:)|(:'\()|(8\))")
RE_HTTP = re.compile("http(s)?://[/\.a-z0-9]+")

def importData(filename):
    reviews = pd.read_csv(filename)
    return reviews

def saveFig(fig, filename):
    fig.savefig(figuresDir + "/" + filename + ".eps", format="eps")

def addBarValues(ax, rects):
    for rect in rects:
        height = np.round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def plotBarGraph(x, y, xlabel, ylabel):
    fig, ax = plt.subplots()

    rects = ax.bar(x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, max(y)*1.1)
    addBarValues(ax, rects)

    return fig

def scoreHistogram(reviews):
    scores = reviews["score"].values
    unique, counts = np.unique(scores, return_counts=True)

    fig = plotBarGraph(unique, counts/1000, xlabel="score", ylabel="Liczność [tys.]")
    fig.show()

    saveFig(fig, "scoreHistogram")

def writeCommonWordsToFile(common_words):
    with open("common_words.txt", 'wb') as f:
        pickle.dump(common_words, f)

def readCommonWords():
    my_list = Counter()
    with open("common_words.txt", 'rb') as f:
        my_list.update(pickle.load(f))
    return my_list
