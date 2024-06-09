import pandas as pd
import streamlit as st
import re
import string
import time
from threading import Thread
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

path = {'file':"Restaurant_Reviews.txt",'logo':"logo-1.jpg"}
rm_neg_wrds = ["not","isn't","wasn't","aren't","weren't","haven't","hadn't","hasn't",'mightn',"mightn't", 'mustn', "mustn't",'needn', "needn't", 'shouldn', "shouldn't","don't",'wasn', 'weren', "won't", 'wouldn', "wouldn't"]
global stp_wrds
stp_wrds = stopwords.words('english')

img, head = st.columns([0.15, 0.85])
img.image(path['logo'])
head.header("Restaurant Review Analysis")

# removing negative words from stopwords
for i in rm_neg_wrds:
    stp_wrds.remove(i)

def text_cleaning(feed):
    feed = feed.lower()
    global feed_splt
    feed_rm_p = re.sub("[^a-zA-Z ']","",feed)
    feed_splt = feed_rm_p.split(" ")
    newdoc = []
    wnt = WordNetLemmatizer()
    for token in feed_splt:
        if token not in stp_wrds:
            newdoc.append(wnt.lemmatize(token))
    return " ".join(newdoc)

def trainmodel():
    df = pd.read_csv(path['file'],delimiter="\t")
    corpus_trn = df.Review.values
    target = df.Liked.values
    mod_corpus_trn = list(map(text_cleaning,corpus_trn))
    global cv
    cv = CountVectorizer(ngram_range=(1,1),min_df=1)
    X = cv.fit_transform(mod_corpus_trn)
    y = target
    global model
    model = MultinomialNB()
    model.fit(X,y)

trainmodel()

def prediction():
    # cleans punctuations/stopwords
    global review,mod_corpus_review
    mod_corpus_review = text_cleaning(review)
    # feature extraction of review given
    global cv,mod_review
    mod_review = cv.transform([mod_corpus_review]).toarray()
    global result
    result = model.predict(mod_review)

#checking for positive/Negative Review
def predict_file(df):
    corpus = df.iloc[:,0].values
    mod_corpus_review = list(map(text_cleaning,corpus))
    global cv
    mod_review = cv.transform(mod_corpus_review).toarray()
    res = model.predict(mod_review)
    global result
    result = []
    for i in res:
        if i==0:
            result.append("Not Liked")
        else:
            result.append("Liked")
    df["Analysis"] = result
    df.columns = ['Review','Analysis']
    st.table(df)


# input your review/feedback
global review
review = st.text_input("**Your Feedback**")

if st.button("Analyse"):
    prediction()
    if result[0]==1:
        st.markdown("""
                    **Liked**
                    :sunglasses:
                    """) 
    elif result[0]==0:
        st.markdown("""
                    **Not Liked**
                    :moon:
                    """)
    

file = st.file_uploader(label="**Bulk Review | Attach your file**")
if st.button("Analyse File"):
    if file:
        if file.name.endswith(".csv") or file.name.endswith(".txt"):
            df = pd.read_csv(file,header=None)
            predict_file(df)
    elif file.name.endswith(".xlsx"):
            df = pd.read_excel(file)
            predict_file(df)
    elif file.name.endswith(".json"):
            df = pd.read_json(file)
            predict_file(df)


