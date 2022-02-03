

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 11:50:49 2022

@author: alexiadeboynes
"""
# import des librairies
import sqlite3
#import mysql.connector
import pymysql
import psycopg2
from sqlalchemy import create_engine
import os
import yaml
import requests
import pandas as pd
import matplotlib.pyplot as plt
from nltk import FreqDist
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import spacy 
from nltk.stem.snowball import FrenchStemmer
from wordcloud import WordCloud
import pandas as pd
import re
#Importing all the libraries to be used
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline    
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from matplotlib.colors import ListedColormap
from sklearn.metrics import precision_score, recall_score, plot_confusion_matrix, classification_report, accuracy_score, f1_score
from sklearn import metrics
'''
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')'''

#loading data

df= pd.read_csv("spam.csv", encoding='latin-1')
#data.info()

# Dropping the 3 unnamed  collumns

to_drop = ["Unnamed: 2","Unnamed: 3","Unnamed: 4"]
df = df.drop(df[to_drop], axis=1)


# Renaming the columns
df.rename(columns = {"v1":"label", "v2":"message"}, inplace = True)
#df.head()


#fonction
'''
def countspam(code):
    n=0
    countsp = 0
    countha = 0
    for line in df['message']: # Parcours des lignes
        x = re.findall(code, line)
        if x:
            print(n, 'findall:', x)#commenter les prints pour de la visibilité
            print(df['label'][n])
            print(df['message'][n])
        
            if df['label'][n]== 'spam':
                countsp +=1
            else:
                countha +=1     
        n+=1
    print('Nombre de Spam :', countsp)
    print('Nombre de Ham :', countha)
    
countspam("prize")
'''

grouped = df.groupby(df.label)
dfspam= grouped.get_group("spam")

#print(dfspam)
dfspam.count()

df3 = dfspam.message.str.lower()
print(df3)

#Palette
cols= ["#E1F16B", "#E598D8"] 
#first of all let us evaluate the target and find out if our data is imbalanced or not
plt.figure(figsize=(12,8))
fg = sns.countplot(x= df["label"], palette= cols)
fg.set_title("Count Plot of Classes", color="#58508d")
fg.set_xlabel("Classes", color="#58508d")
fg.set_ylabel("Number of Data points", color="#58508d")



#Adding a column of numbers of charachters,words and sentences in each msg
df["No_of_Characters"] = df["message"].apply(len)
df["No_of_Words"]=df.apply(lambda row: nltk.word_tokenize(row["message"]), axis=1).apply(len)
df["No_of_sentence"]=df.apply(lambda row: nltk.sent_tokenize(row["message"]), axis=1).apply(len)

df.describe().T

plt.figure(figsize=(12,8))
fg = sns.pairplot(data=df, hue="label",palette=cols)
plt.show(fg)


#Dropping the outliers. 
df = df[(df["No_of_Characters"]<350)]
df.shape

def binarycolumn(code, namecol):
    n=0
    df.insert(2, namecol, "0", allow_duplicates=False)
    for line in df['message']: # Parcours des lignes
        x = re.findall(code, line)
        if x:
            df[namecol].array[n] = len(x)
        else:
            df[namecol].array[n] = 0
        n+=1   

binarycolumn("(?i)(free)", "free")
binarycolumn("[0-9]{10}", "phone")
binarycolumn("(?i)(call)", "call")
binarycolumn("(?i)(mobile)", "mobile")
binarycolumn("(?i)(text)", "text")
binarycolumn("(?i)(txt)", "txt")
binarycolumn("(?i)(now)", "now")
binarycolumn("(?i)(please call)", "please call")
binarycolumn("(?i)(reply)", "reply")
binarycolumn("(?i)(prize)", "prize")
binarycolumn("(?i)(cash)", "cash")


print(df)


# Defining a function to clean up the text
def Clean(Text):
    sms = re.sub('[^a-zA-Z]', ' ', Text) #Replacing all non-alphabetic characters with a space
    sms = sms.lower() #converting to lowecase
    sms = sms.split()
    sms = ' '.join(sms)
    return sms


df["Clean_Text"] = df["message"].apply(Clean)
#Lets have a look at a sample of texts after cleaning
print("\033[1m\u001b[45;1m The First 5 Texts after cleaning:\033[0m",*df["Clean_Text"][:5], sep = "\n")


df["Tokenize_Text"]=df.apply(lambda row: nltk.word_tokenize(row["Clean_Text"]), axis=1)

print("\033[1m\u001b[45;1m The First 5 Texts after Tokenizing:\033[0m",*df["Tokenize_Text"][:5], sep = "\n")

# Removing the stopwords function
def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    filtered_text = [word for word in text if word not in stop_words]
    return filtered_text

df["Nostopword_Text"] = df["Tokenize_Text"].apply(remove_stopwords)

print("\033[1m\u001b[45;1m The First 5 Texts after removing the stopwords:\033[0m",*df["Nostopword_Text"][:5], sep = "\n")


lemmatizer = WordNetLemmatizer()
# lemmatize string
def lemmatize_word(text):
    #word_tokens = word_tokenize(text)
    # provide context i.e. part-of-speech
    lemmas = [lemmatizer.lemmatize(word, pos ='v') for word in text]
    return lemmas

df["Lemmatized_Text"] = df["Nostopword_Text"].apply(lemmatize_word)
print("\033[1m\u001b[45;1m The First 5 Texts after lemitization:\033[0m",*df["Lemmatized_Text"][:5], sep = "\n")



#Creating a corpus of text feature to encode further into vectorized form
corpus= []
for i in df["Lemmatized_Text"]:
    msg = ' '.join([row for row in i])
    corpus.append(msg)
    
corpus[:5]
print("\033[1m\u001b[45;1m The First 5 lines in corpus :\033[0m",*corpus[:5], sep = "\n")



#Changing text data in to numbers. 
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()
#Let's have a look at our feature 
X.dtype

print(X.shape)


#Setting values for labels and feature as y and X(we already did X in vectorizing...)
y = df["label"] 
# Splitting the testing and training sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Testing on the following classifiers
classifiers = [MultinomialNB(), 
               RandomForestClassifier(),
               KNeighborsClassifier(), 
               SVC()]
for cls in classifiers:
    cls.fit(X_train, y_train)

# Dictionary of pipelines and model types for ease of reference
pipe_dict = {0: "NaiveBayes", 1: "RandomForest", 2: "KNeighbours",3: "SVC"}


# Crossvalidation 
for i, model in enumerate(classifiers):
    cv_score = cross_val_score(model, X_train,y_train,scoring="accuracy", cv=10)
    print("%s: %f " % (pipe_dict[i], cv_score.mean()))

    


from sklearn.neighbors import KNeighborsClassifier
# Initialisation with the choice of k = 3
KNN_classifier = KNeighborsClassifier(n_neighbors=5)
classifiertest(KNN_classifier, X, y)

from sklearn.preprocessing import LabelEncoder
label_encod = LabelEncoder()
dfe =label_encod.fit_transform(df['label'])
y = dfe
X =df[['length','free','phone','prize','give','devise']]




from wordcloud import WordCloud
spam_wordcloud = WordCloud(width=600, height=400).generate(" ".join(dfspam['message']))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(spam_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()




ham_wordcloud = WordCloud(width=600, height=400).generate(" ".join(df['message']))
plt.figure( figsize=(10,8), facecolor='k')
plt.imshow(ham_wordcloud)
plt.axis("off")
plt.tight_layout(pad=0)
plt.show()






