import streamlit as st
import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import OrdinalEncoder

stpwrds = stopwords.words("english")
path = {'trainfile':'News_Category_Dataset_v3.json','logo':'logo-1.jpg'}

img,head = st.columns([0.15,0.85])
img.image(path['logo'])
head.header("**Restaurant News Recommendations**")

def text_cleaning(heading):
    heading = heading.lower()
    head = re.sub("[^a-zA-Z ']","",heading)
    head_splt = head.split()
    newhead = []
    wnl =WordNetLemmatizer()
    for token in head_splt:
        if token not in stpwrds:
            newhead.append(wnl.lemmatize(token))
    return " ".join(newhead)

def trainmodel():
    global df
    df = pd.read_json(path['trainfile'],lines=True)
    df1 = df[['headline','category']]
    df1 = df1[df.category.isin(['ENTERTAINMENT','WELLNESS','POLITICS'])]
    ord = OrdinalEncoder()
    yt = ord.fit_transform(df[['category']])
    global news_category
    news_category = ord.categories_
    corpus = df.headline.values
    corpus_mod = list(map(text_cleaning,corpus))
    global cv
    cv = CountVectorizer()
    Xt = cv.fit_transform(corpus_mod)
    global model
    model = MultinomialNB()
    model.fit(Xt,yt)

def prediction(sample):
    st = text_cleaning(sample)
    global cv
    stt = cv.transform([st]).toarray()
    p = model.predict(stt)[0]
    global cat
    cat = news_category[0][int(p)]

def show_similar():
    global df
    news = df[(df.category==cat)]
    news = news.sort_values(by=df.date,ascending=True).head()
    st.markdown("""
    **Similar News**
    """)
    st.table(news.head())

trainmodel()

h = st.text_input("**Your News Headline:**")
if st.button("Get"):
    prediction(h)
    st.markdown(f"""
    :newspaper:
    **{cat}**
    :newspaper:""")
    show_similar()


