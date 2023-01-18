import sklearn
from sklearn import svm
import numpy as np
import pandas as pd
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def remove_stop_words(data, stop_words):
    datax = data.copy()
    for i in range(data.size):
        word = data[i].split()
        for j in word.copy():
            if(j.lower() in stop_words):
                word.remove(j)
        for j in range(len(word)):
            word[j] = word[j].lower()
        datax[i] = " ".join(word)
    return datax

def stem_words(data, stem):
    datax = data.copy()
    for i in range(data.size):
      word = data[i].split()
      word = list(stem(x) for x in word)
      datax[i] = " ".join(word)
    return datax

def remove(data):
    datax = data.copy()
    for i in range(data.size):
      datax[i] = ''.join(char for char in data[i] if char.isalnum() or char == ' ')
    return datax

def dataset(data):
    vc = 0.5
    
    stop_words = set(stopwords.words('english'))
    
    size = data["text"].size
    
    data_title = data["title"]
    data_text = data["text"]
    data_date = data["date"]
    data_truth = data["truth"]

    data_title = remove_stop_words(data_title, stop_words)
    data_text = remove_stop_words(data_text, stop_words)
    print("=Stop words removed")
    porter = PorterStemmer()
  
    data_title = stem_words(data_title, porter.stem)
    data_text = stem_words(data_text, porter.stem)
    print("=Stemed")

    data_title = remove(data_title)
    data_text = remove(data_text)
    print("=No symbols")
    
    tokens = nltk.word_tokenize(' '.join(data_title) + ' ' + ' '.join(data_text))
    set_all = set(i for i in tokens if len(i) > 1)
    tokens = [i for i in tokens if i in set_all]
    freq_tokens = nltk.FreqDist(tokens)
    vocab = freq_tokens.most_common(2500)

    return (data_title, data_text, data_date, data_truth), vocab

def transform_bag(data, vocab):
    a = np.zeros((data.size, len(vocab)))
    for i in range(data.size):
      word = data[i].split()
      for j in word:
        if j in vocab:
          a[i][vocab.index(j)] += 1
    for i in range(len(a)):
        data[i] = a[i]
    return a

def split(data):
    size = int(data.shape[0]*0.8)
    return (data[:size], data[size:])

false = pd.read_csv("Fake.csv")
false["truth"] = 0
true = pd.read_csv("True.csv")
true["truth"] = 1
print("Data loaded")
news_all = pd.concat([false, true])
news_shuffled = sklearn.utils.shuffle(news_all)
news_reindexed = news_shuffled.reset_index(drop = True)
print("Data merged")
news_cut = news_reindexed.sample(frac = 0.3)
news_cut = news_cut.reset_index(drop = True)
data, vocab = dataset(news_cut)
print("Data edited")
vocab = np.array(vocab).T[0].tolist()

xx = transform_bag(data[0], vocab)
xy = transform_bag(data[1], vocab)
print("Transform baged")

full = pd.DataFrame()
x = np.concatenate((xx,xy), axis=1)
y = data[3]
size = y.size
full_x = split(x)
full_y = split(y)
big_x = split(x[:int(0.2*size)])
big_y = split(y[:int(0.2*size)])
medium_x = split(x[int(0.2*size):int(0.25*size)])
medium_y = split(y[int(0.2*size):int(0.25*size)])
small_x = split(x[int(0.25*size):int(0.26*size)])
small_y = split(y[int(0.25*size):int(0.26*size)])
tiny_x = split(x[int(0.26*size):int(0.262*size)])
tiny_y = split(y[int(0.26*size):int(0.262*size)])
print("sampled")

svcF = svm.SVC(C=1, kernel='rbf')
svcF.fit(full_x[0], full_y[0])
print("Traning accuracy for full SVM: {}".format(svcF.score(full_x[0], full_y[0])))

svcB = svm.SVC(C=1, kernel='rbf')
svcB.fit(big_x[0], big_y[0])
print("Traning accuracy for big SVM: {}".format(svcB.score(big_x[1], big_y[1])))

svcM = svm.SVC(C=1, kernel='rbf')
svcM.fit(medium_x[0], medium_y[0])
print("Traning accuracy for medium SVM: {}".format(svcM.score(medium_x[1], medium_y[1])))

svcS = svm.SVC(C=1, kernel='rbf')
svcS.fit(small_x[0], small_y[0])
print("Traning accuracy for small SVM: {}".format(svcS.score(small_x[1], small_y[1])))

svcT = svm.SVC(C=1, kernel='rbf')
svcT.fit(tiny_x[0], tiny_y[0])
print("Traning accuracy for tiny SVM: {}".format(svcT.score(tiny_x[1], tiny_y[1])))

print("No text")
no_text_x = big_x[0][:,:2500], big_x[1][:,:2500]
svcT = svm.SVC(C=1, kernel='rbf')
svcT.fit(no_text_x[0], big_y[0])
print("Traning accuracy for no text SVM: {}".format(svcT.score(no_text_x[1], big_y[1])))

print("No title")
no_title_x = big_x[0][:,2500:], big_x[1][:,2500:]
svcS = svm.SVC(C=1, kernel='rbf')
svcS.fit(no_title_x[0], big_y[0])
print("Traning accuracy for no title SVM: {}".format(svcS.score(no_title_x[1], big_y[1])))
