import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
review = pd.read_csv("reviews.csv")

 
X = review.review
y = review.polarity
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.6, random_state = 1)


vector = CountVectorizer(stop_words = 'english',lowercase=False)

vector.fit(X_train)
X_transformed = vector.transform(X_train)
X_transformed.toarray()

X_test_transformed = vector.transform(X_test)


naivebayes = MultinomialNB()
naivebayes.fit(X_transformed, y_train)


saved_model = pickle.dumps(naivebayes)
s = pickle.loads(saved_model)

st.header('NB Sentiment Analyser')
input = st.text_area('Enter your text:', value="")
if st.button("Analyse"):
    vec = vector.transform([input]).toarray()
    st.write('Label:',str(list(s.predict(vec))[0]).replace('0', 'NEGATIVE').replace('1', 'POSITIVE'))