import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb

dataset = pd.read_csv(r'C:\Users\vinay\Downloads\Restaurant_Reviews.tsv',delimiter='\t')

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

corpus = []

for i in range(0,1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    

    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]   
    review = ' '.join(review)
    corpus.append(review)
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()

y = dataset.iloc[:,1]
print(y)

# lets apply machine learning model
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2,random_state=0)

#from sklearn.tree import DecisionTreeClassifier
#classifier = DecisionTreeClassifier()
#classifier.fit(x_train,y_train)

#from sklearn.ensemble import RandomForestClassifier
#classifier = RandomForestClassifier()
#classifier.fit(x_train,y_train)

#from xgboost import XGBClassifier # Error
#classifier = XGBClassifier()
#classifier.fit(x_train,y_train)

#from sklearn.neighbors import KNeighborsClassifier
#classifier = KNeighborsClassifier()
#classifier.fit(x_train,y_train)

#from sklearn.svm import SVC
#classifier = SVC()
#classifier.fit(x_train,y_train)

from sklearn.naive_bayes import MultinomialNB
classifier = MultinomialNB()
classifier.fit(x_train,y_train)

#from sklearn.naive_bayes import GaussianNB
#classifier = GaussianNB()
#classifier.fit(x_train,y_train)

#from sklearn.linear_model import LogisticRegression
#classifier = LogisticRegression()
#classifier.fit(x_train,y_train)

#from sklearn.ensemble import GradientBoostingClassifier
#classifier = GradientBoostingClassifier()
#classifier.fit(x_train, y_train)

#from lightgbm import LGBMClassifier
#classifier = LGBMClassifier()
#classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test,y_pred)
print(ac)


bias = classifier.score(x_train,y_train)
print(bias)

variance = classifier.score(x_test,y_test)
print(variance)















