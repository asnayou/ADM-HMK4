import numpy as np
import pandas as pd
import sklearn
import random
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import re
import matplotlib.pyplot as plt

REGEX_SPECIAL_CHARACTERS = ['+', '-', '*', '|', '&', '[', ']', '(', ')', '{',
                '}', '^', '?', '.', '$', ',', ':', '=', '#', '!', '<',' ','"','\\','/',"'","%","~",'\n','>',';']

class Tokenizer:
        def __init__(self, separators):
                separators = ['\\'+sep if sep in REGEX_SPECIAL_CHARACTERS else sep for sep in separators]
                self._regex = '[' + ''.join(separators) + ']+'

        def tokenize(self, s):
                return [t for t in re.split(self._regex, s) if t != '']

def my_kmeans(X,n_clusters):

    #Initialisation : random selection of the cluster's centers among datas
    clusters_centers=[]
    l=[i for i in range(X.shape[0])]
    while len(clusters_centers)<n_clusters:
        clusters_centers.append(X[np.random.choice(l)])
    #print("First centers are {}".format(clusters_centers) + "\n")

    epsilon=3
    iterations=0
    while epsilon>10e-6: #stopping criterion

        #creation of the distance matrix --> matrix with storing distances between centers in rows and points in columns
        tab=[]
        for i in range(len(clusters_centers)):
            dist=[]
            for j in range(X.shape[0]):
                dist.append(np.linalg.norm(clusters_centers[i]-X[j]))

            tab.append(dist)
        tab=np.array(tab)
        #print(tab)

        #pooling of points to their nearest center
        L=[]
        for i in range(tab.shape[1]): #loop on each points
            a=10e6
            ide=0
            for j in range(tab.shape[0]): #loop on each center
                if tab[j][i]<a:    #looking for the smallest distance between center j and point i
                    a=tab[j][i]
                    ide=j          #storing indice of the center the closest
            L.append(ide)
        #print(str(L)+"\n")


        #computing of new clusters centers with center of gravity

        dic={}
        for i in range(len(clusters_centers)): #create a dictionnary with number representing the id of the clusters_centers
            dic[i]=[]
        for i in range(len(L)):
            dic[L[i]].append(X[i])

        temp=[]
        for i in (dic.keys()):       #we change the value of clusters_centers with the gravity centers
            temp.append(gravity(dic[i]))

        #stoping criterion computation
        epsilon=abs((np.array(clusters_centers)-np.array(temp)).sum())
        print("Epsilon value at iteration number {} is {}.".format(iterations, epsilon))

        clusters_centers=temp
        iterations+=1

    print("\n The process finally required {} iterations.\n".format(iterations))
    return clusters_centers, dic

def gravity(l):
    x,y=0,0
    for i in range(len(l)):
        x+=l[i][0]
        y+=l[i][1]
    x=x/len(l)
    y=y/len(l)
    return np.array([x,y])

def remove_stopwords(s):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(s)
    return ' '.join([w for w in tokens if w not in stop_words])

def stemming(s):
    ps = PorterStemmer()
    tokens = word_tokenize(s)
    return ' '.join([ps.stem(w) for w in tokens])

def tfidf_vectorizer(df):
    tfidf_vectorizer=TfidfVectorizer()
    tfidf_new=tfidf_vectorizer.fit_transform(df["Text"])
    return tfidf_new

def svd_reduction(X):
    svd = TruncatedSVD(n_components=100, n_iter=5) #i took those parameters regarding what was written in the SVD manual
    A=svd.fit_transform(X)
    return A

def clean(s, stop_list):
    tokens=tokenizer.tokenize(s)
    return ' '.join([w for w in tokens if w not in stop_list])