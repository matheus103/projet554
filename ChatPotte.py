import csv
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from verstack.stratified_continuous_split import scsplit
from os import path
import time
import pickle as pk
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
import gensim
import re
import string
from nltk.corpus import stopwords
import tensorflow as tf
import keras
from wordcloud import WordCloud, STOPWORDS
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Dropout, Bidirectional, GRU
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Input
from keras.layers import concatenate
from keras.optimizers import Adam
from keras.preprocessing import text, sequence
import psutil

plt.style.use('dark_background')


## Loading

def load_it(pickone):
    print('Loading data...', end='')
    X_train_full_in = pd.read_csv("data/train.csv")
    X_train_in, X_test_in, y_train_in, y_test_in = \
        scsplit(X_train_full_in,
                X_train_full_in['retweet_count'],
                stratify=X_train_full_in['retweet_count'],
                train_size=0.7,
                test_size=0.3)
    X_train_in = X_train_in.drop(['retweet_count'], axis=1)
    X_test_in = X_test_in.drop(['retweet_count'], axis=1)
    X_eval_in = pd.read_csv("data/evaluation.csv")
    y_train_full_in = X_train_full_in['retweet_count']
    X_train_full_in = X_train_full_in.drop(['retweet_count'], axis=1)
    print('done')

    if pickone == 'test':
        return X_train_in, y_train_in, X_test_in, y_test_in

    if pickone == 'eval':
        return X_train_full_in, y_train_full_in, X_eval_in

    if pickone == 'both':
        return X_train_in, y_train_in, X_test_in, y_test_in, X_train_full_in, y_train_full_in, X_eval_in

def to_numeric(X):
    for col in X.columns:
        X[col] = pd.to_numeric(X[col])


## Some functions for adding features
def cut_nan(X, col_name):  # On remplace les nan par des ''
    tab = pd.DataFrame.to_numpy(X[col_name])
    for i in range(len(tab)):
        if not (isinstance(tab[i], str)):
            tab[i] = ""
    X[col_name] = tab


def rass(u):  # Rassifie un string
    v = u.replace('twitter.com/i/web/status/1â€¦', 'twitter')
    for c in ['/', '.', '_', '-', ',']:  # séparateurs
        v = v.replace(c, ' ')
    for c in ['1â€¦', 'fâ€¦', 'â€¦']:  # à tej
        v = v.replace(c, ' ')
    return v


##
def remove_urls(s):
    return re.sub(r'http\S+', ' ', s)


def remove_mentions(s):
    return re.sub(r'@\S+', ' ', s)


def remove_hashtags(s):
    return re.sub(r'#\S+', ' ', s)


def remove_special(s):
    return re.sub(r'[^a-zA-Z0-9 ]', r' ', s)


def remove_emoji(s):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r' ', s)


stop = set(stopwords.words('english'))
punctuation = list(string.punctuation)
stop.update(punctuation)
def remove_stopwords(s):
    final_text = []
    for m in s.split():
        if m.strip().lower() not in stop:
            final_text.append(m.strip())
    return " ".join(final_text)


def clean_text(s):
    s = remove_urls(s)
    s = remove_mentions(s)
    s = remove_emoji(s)
    s = remove_hashtags(s)
    s = re.sub(r'&amp', r' ', s)
    s = remove_special(s)
    s = remove_stopwords(s)
    s = s.lower()
    return s


def ras_col(X, col_name):
    tab = pd.DataFrame.to_numpy(X['urls'])
    for i in range(len(tab)):
        tab[i] = rass(tab[i])
    X[col_name] = tab


def tfidf_col(_X_train, _X_test, col_name, n_features=100):
    cut_nan(_X_train, col_name)
    cut_nan(_X_test, col_name)

    if col_name == 'urls':
        ras_col(_X_train, 'urls')
        ras_col(_X_test, 'urls')

    vectorizer = TfidfVectorizer(max_features=n_features, stop_words='english')
    utv_train = vectorizer.fit_transform(_X_train[col_name])
    utv_test = vectorizer.transform(_X_test[col_name])

    for i in range(n_features):
        _X_train[col_name[0:2] + 'tv' + str(i)] = utv_train[:, i].toarray()
        _X_test[col_name[0:2] + 'tv' + str(i)] = utv_test[:, i].toarray()


def binarize_user_verified(X):
    X['user_verified'] = pd.DataFrame.to_numpy(X['user_verified']) * 1


def add_text_len(X):  # On pourrait ajouter le nombre de mentions, etc..
    tab = pd.DataFrame.to_numpy(X['text'])
    tab2 = np.zeros(tab.shape[0])
    for i in range(len(tab)):
        tab2[i] = len(tab[i])
    X['text_len'] = tab2


def add_number_items(X, colname):
    cut_nan(X, colname)
    tab = pd.DataFrame.to_numpy(X[colname])
    tab2 = np.zeros_like(tab)
    for i in range(len(tab)):
        tab2[i] = len(tab[i].split(','))
    X[colname+'_nItems'] = tab2


def standardize(X):
    for col_name in X.columns:
        tab = np.array(X[col_name])
        X[col_name] = (tab - tab.mean()) / tab.std()


def drop_it(X, col_name):
    X.drop([col_name], axis=1, inplace=True)


def get_weight_matrix(model, vocab, EMBEDDING_DIM):
    # total vocabulary size plus 0 for unknown words
    vocab_size = len(vocab) + 1
    # define weight matrix dimensions with all 0
    weight_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
    # step vocab, store vectors using the Tokenizer's integer mapping
    for word, i in vocab.items():
        weight_matrix[i] = model[word]
    return weight_matrix


def generate_w2v_model(_X_train, _X_eval, word_dim):
    words = []
    for i in _X_train.text.values:
        words.append(i.split())
    for i in _X_eval.text.values:
        words.append(i.split())
    w2v_model = gensim.models.Word2Vec(sentences=words, size=word_dim, window=5, min_count=1)
    tokenizer = text.Tokenizer(num_words=len(w2v_model.wv.vocab))
    tokenizer.fit_on_texts(words)
    return tokenizer, w2v_model


def model_to_tok(_X, _tokenizer):
    words = []
    for i in _X.text.values:
        words.append(i.split())
    tokenized = _tokenizer.texts_to_sequences(words)
    x = sequence.pad_sequences(tokenized, maxlen=20)
    return x


def word_to_vect(_X_train, _X_eval, word_dim):
    tokenizer, w2v_model = generate_w2v_model(_X_train, _X_eval, word_dim)
    x_train, x_test = model_to_tok(_X_train, tokenizer), model_to_tok(_X_eval, tokenizer)
    embedding_vectors = get_weight_matrix(w2v_model, tokenizer.word_index, word_dim)
    word_vect_train = np.zeros((_X_train.shape[0], word_dim))
    word_vect_test = np.zeros((_X_eval.shape[0], word_dim))
    for i in range(_X_train.shape[0]):
        word_vect_train[i] = np.sum(embedding_vectors[x_train[i]], axis=0)
    for i in range(_X_eval.shape[0]):
        word_vect_test[i] = np.sum(embedding_vectors[x_test[i]], axis=0)

    for j in range(word_dim):
        _X_train['w2v'+str(j)] = word_vect_train[:, j]
        _X_eval['w2v' + str(j)] = word_vect_test[:, j]
    return None


def boost_it(_Xs, word_dim=200, tok_dim=10):       # _Xs = [_X_train, _X_test]
    Xs = []
    for X in _Xs:
        X['text'] = X['text'].apply(clean_text)
        Xs.append(pd.DataFrame.copy(X))
    print('Adding features...', end='')

    col_names = ['urls', 'hashtags', 'user_mentions']
    for X in Xs:
        add_text_len(X)
        for col_name in col_names:
            add_number_items(X, col_name)

    col_names = ['hashtags', 'user_mentions']           # Rien à battre des urls
    if tok_dim > 0:
        for col_name in col_names:
            tfidf_col(Xs[0], Xs[1], col_name, n_features=tok_dim)
            if len(Xs) == 4:
                tfidf_col(Xs[2], Xs[3], col_name, n_features=tok_dim)

    if word_dim > 0:
        word_to_vect(Xs[0], Xs[1], word_dim)
        if len(Xs) == 4:
            word_to_vect(Xs[2], Xs[3], word_dim)

    for X in Xs:
        binarize_user_verified(X)
        to_drop = ['id', 'text', 'urls', 'hashtags', 'user_mentions']
        for col_name in to_drop:
            drop_it(X, col_name)
        standardize(X)
        to_numeric(X)
    print('done')
    return Xs


## Test
def test_it(_X_train, _y_train, _X_test, _y_test, regressor, eval_err=True):
    print('Starting training...', end='')
    regressor.fit(_X_train, _y_train)
    _y_pred = regressor.predict(_X_test)
    print('done')
    if eval_err:
        print("Prediction error:", mae(_y_test, _y_pred))
    return _y_pred


def test_it_bin(_X_train, _y_train, _X_test, _y_test, classifier, eval_err=True):
    print('Starting learning...', end='')
    classifier.fit(_X_train, bin_y(_y_train))
    _z_pred = classifier.predict(_X_test)
    print('done')
    if eval_err:
        print("Prediction error:", ratio(bin_y(_y_test), _z_pred))
    return _z_pred


## Evaluation
def eval_it(_X_train, _y_train, _X_eval, _X_eval_id, regressor, save=True):
    print('Evaluation...', end='')
    regressor.fit(_X_train, _y_train)
    _y_pred = regressor.predict(_X_eval)
    print('done')
    if save:
        save_it(_X_eval_id, _y_pred)
    return _y_pred


def eval_it_bin(_X_train, _y_train, _X_eval, _X_eval_id, classifier, save=True):
    print('Evaluation...', end='')
    classifier.fit(_X_train, bin_y(_y_train))
    _z_pred = classifier.predict(_X_eval)
    print('done')
    if save:
        save_it(_X_eval_id, _z_pred)
    return _z_pred


def save_it(_X_eval, _y_pred):
    i = 1
    while path.exists("submissions/submission" + str(i) + ".txt"):
        i += 1
    print('Saving in', "submissions/submission" + str(i) + ".txt...", end='')
    with open("submissions/submission" + str(i) + ".txt ", 'w') as f:
        writer = csv.writer(f)
        writer.writerow(["TweetID", "NoRetweets"])
        for index, prediction in enumerate(_y_pred):
            writer.writerow([str(_X_eval['id'].iloc[index]), str(int(prediction))])
    print('done')

## Des fonctions en plus
def mae(y1, y2):
    return np.mean(np.abs(y1 - y2))


def cut_y(y, eps):
    y_in = np.copy(y)
    for i in range(len(y_in)):
        if y_in[i] <= eps:
            y_in[i] = 0
    return y_in


def cut_y_inv(y, eps):
    y_in = np.copy(y)
    for i in range(len(y_in)):
        if y_in[i] > eps:
            y_in[i] = 300000
    return y_in


def bin_y(y_base):
    y_in = np.copy(y_base)
    for i in range(len(y_in)):
        if y_in[i] < 0.5:
            y_in[i] = 0
        else:
            y_in[i] = 1
    return y_in


def ratio(a, b):
    return np.sum(a == b) / a.shape[0]


def ratio_faux_positif(a, b):
    return np.sum((a == b) * b) / np.sum(b)


def ratio_faux_negatif(a, b):
    return np.sum((a == b) * (1 - b)) / np.sum(1 - b)


def display_ratios(a, b):
    print('Précision :', ratio(a, b), '\nTp :', ratio_faux_positif(a, b), '\nTn :', ratio_faux_negatif(a, b))

def hype(_X_test, _y_test, _z_pred, start=0, dx=5):
    print('Quelques exemples de tweets qui ont buzzé :')
    for i in np.array(_X_test['id'])[_z_pred == 1][start:start + dx]:
        print(_X_test.loc[i].text)
        print('Nombre de tweets réel :', _y_test.loc[i])

def display_history(history):
    epochs = [i for i in range(len(history.history['loss']))]
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.plot(epochs, train_loss, 'go-', label='Training Loss')
    plt.plot(epochs, val_loss, 'ro-', label='Testing Loss')
    plt.title('Training & Testing Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()
    return None

##

##
# plt.figure(figsize=(20, 20))  # Text that is Not Sarcastic
# wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(X_train[y_train > 0].text))
# plt.title('Tweets retweetés')
# plt.imshow(wc, interpolation='bilinear')
#
# ##
# plt.figure(figsize=(20, 20))  # Text that is Not Sarcastic
# wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(X_train[y_train == 0].text))
# plt.title('Tweets non retweetés')
# plt.imshow(wc, interpolation='bilinear')

##

def mu():
    import psutil
    return psutil.virtual_memory().percent
##


