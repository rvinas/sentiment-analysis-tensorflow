from tsa.data_manager import DataManager
from gensim import models, corpora
import pandas as pd
import re
from pyLDAvis import gensim, display

"""
dm = DataManager(data_dir='data/Kaggle',
                 stopwords_file='data/stopwords.txt',
                 sequence_len=None,
                 test_size=0.1,
                 val_samples=0,
                 n_samples=None,
                 random_state=None)
"""

toIgnore_expr = [r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&amp;+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+',  # url
                 r'&[a-z]+;'  # html entities
                 ]
all_toIgnore = '|'.join(toIgnore_expr)


def read_stopwords():
    """
    :return: Stopwords list
    """
    with open('data/stopwords.txt', mode='r') as f:
        stopwords = f.read().splitlines()
    return stopwords


def remove_waste(text):
    return re.sub(all_toIgnore, " ", text)


def tokenize(sentence):
    stopw = read_stopwords()
    return [word for word in re.findall(r"[#|@|A-Z|a-z]\w*", remove_waste(sentence))
            if len(word) > 1 and word[0] != "#" and word[0] != "@" and word.lower() not in stopw]


data = pd.read_csv('data/Kaggle/data.csv')
samples = data.as_matrix(columns=['SentimentText'])[:, 0]

tokenized = [tokenize(txt.lower()) for i, txt in enumerate(samples)]

print(tokenized)

dictionary = corpora.Dictionary(tokenized)
corpus = [dictionary.doc2bow(doc) for doc in tokenized]
K = 30

lda = models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=K, passes=10)
for topic, words in lda.show_topics(formatted=False, num_topics=30, num_words=2):
    print("Topic: {0}, words: {1}".format(topic, words))

vis_data = gensim.prepare(lda, corpus, dictionary)
display(vis_data)
