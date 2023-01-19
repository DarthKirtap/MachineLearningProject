import sklearn
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
import datetime
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import time
import matplotlib.pyplot as plt

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

    return (data_title, data_text, data_date, data_truth), vocab, freq_tokens

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

def cal(data, fr, size):
    datax = split(data[0][:int(fr*size)])
    datay = split(data[1][:int(fr*size)])
    start_time = time.time()
    svcF = svm.SVC(C=1, kernel='rbf')
    svcF.fit(datax[0], datay[0])
    return (time.time() - start_time, svcF.score(datax[1], datay[1])*100, fr*size)

start_time = time.time()
false = pd.read_csv("Fake.csv")
false["truth"] = 0
true = pd.read_csv("True.csv")
true["truth"] = 1
print("Data loaded")
news_all = pd.concat([false, true])
news_shuffled = sklearn.utils.shuffle(news_all)
news_cut = news_shuffled.sample(frac = 0.3)
news_cut = news_cut.reset_index(drop = True)
print("Data merged")

data, vocab, freq = dataset(news_cut)
vocab = np.array(vocab).T[0].tolist()
print("Data edited")

xx = transform_bag(data[0], vocab)
xy = transform_bag(data[1], vocab)
print("Transform baged")

full = pd.DataFrame()
x = np.concatenate((xx,xy), axis=1)
y = data[3]
size = y.size
big_x = split(x[:int(0.1*size)])
big_y = split(y[:int(0.1*size)])
print("Sampled")
print("--- %s seconds ---" % (time.time() - start_time))

a = []
for i in range(11):
    a.append(cal((x,y), 0.005 + 0.0009*2**i, size))

a = np.array(a)
plt.plot(a[:,2], a[:,0])
plt.plot(a[:,2], a[:,1])
plt.xlabel("Počet vzorkov")
plt.ylabel("Cas vypoctov (s)/Presnost klasifikacie (%)")
plt.xscale('log')
plt.grid()
plt.figure()
plt.plot(a[:,2], a[:,1])
plt.xlabel("Počet vzorkov")
plt.ylabel("Presnost klasifikacie")
plt.xscale('log')
plt.grid()
plt.figure()
plt.plot(a[:,2], a[:,0])
plt.xlabel("Počet vzorkov")
plt.ylabel("Cas vypoctov (s)")
plt.xscale('log')
plt.grid()

print("------------------")
start_time = time.time()
svcB = svm.SVC(C=1, kernel='rbf')
svcB.fit(big_x[0], big_y[0])
print("Traning accuracy for all SVM: {}".format(svcB.score(big_x[1], big_y[1])))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("No text")
no_text_x = big_x[0][:,:2500], big_x[1][:,:2500]
svcT = svm.SVC(C=1, kernel='rbf')
svcT.fit(no_text_x[0], big_y[0])
print("Traning accuracy for no text SVM: {}".format(svcT.score(no_text_x[1], big_y[1])))
print("--- %s seconds ---" % (time.time() - start_time))

start_time = time.time()
print("No title")
no_title_x = big_x[0][:,2500:], big_x[1][:,2500:]
svcS = svm.SVC(C=1, kernel='rbf')
svcS.fit(no_title_x[0], big_y[0])
print("Traning accuracy for no title SVM: {}".format(svcS.score(no_title_x[1], big_y[1])))
print("--- %s seconds ---" % (time.time() - start_time))

rfc = RandomForestClassifier(n_estimators=6, max_depth=6)
rfc.fit(big_x[0], big_y[0])
rfc.score(big_x[1], big_y[1])

for i in range(len(rfc.estimators_)):
    dot_data = tree.export_graphviz(rfc.estimators_[i],
                                    out_file="tree" + str(i) + ".dot",
                                    feature_names=vocab*2,
                                    class_names=['false', 'true'],
                                    rounded=True, filled=True)

plt.show()


