import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

img, head = st.columns([0.2,0.8])
img.image("logo-1.jpg")
head.header("Restaurant Spam-Ham Detection")

selected = option_menu(
    menu_title=None,
    options = ["Single Message", "Mass Checker"],
    icons = ["chat-left-dots","funnel"],
    default_index=0,
    orientation = "horizontal",
    styles = {
        "nav-link-selected":{"background-color":"blue"}
    }
)

stpwrds = stopwords.words("english")
def text_cleaning(msg):
    msg = msg.lower()
    msg = re.sub("[^a-zA-Z ']","",msg)
    msg_splt = msg.split()
    mod_msg = []
    wnl = WordNetLemmatizer()
    for token in msg_splt:
        if token not in stpwrds:
            mod_msg.append(wnl.lemmatize(token))
    return " ".join(mod_msg)

def trainmodel():
    df = pd.read_csv("spam_ham.txt",delimiter="\t")
    corpus = df.iloc[:,-1].values
    target = df.iloc[:,0].values
    mod_corpus = list(map(text_cleaning,corpus))
    global cv
    cv = CountVectorizer()
    Xt = cv.fit_transform(mod_corpus).toarray()
    yt = target
    global model
    model = MultinomialNB()
    model.fit(Xt,yt)

def prediction(sample):
    global cv,model
    ms = text_cleaning(sample)
    mst = cv.transform([ms]).toarray()
    result = model.predict(mst)
    return result[0]

def massprediction(df_s):
    corpus_s = df_s.iloc[:,-1].values
    global cv,model
    ms = list(map(text_cleaning,corpus_s))
    mst = cv.transform([ms]).toarray()
    result = model.predict(mst)
    df_s["Result"] = result
    df_s.columns = ['Message','Result']
    st.table(df_s)

trainmodel()

if selected == "Single Message":
    sample = st.text_input("Your Message:")
    if st.button("Detect"):
        if sample!="":
            res = prediction(sample=sample)
            st.success(res.upper())
        else:
            st.warning("Please! Enter your message to check.",icon="⚠️")
            
if selected =="Mass Checker":
    file = st.file_uploader("**Upload your file to check all messages for SPAM.**")
    if file:
        st.success("File Successfully uploaded.")
    if st.button("Mass Check"):
        if file:
            if file.name.endswith(".csv") or file.name.endswith(".txt"):
                df = pd.read_csv(file,header=None)
                massprediction(df)
        elif file.name.endswith(".xlsx"):
                df = pd.read_excel(file)
                massprediction(df)
        elif file.name.endswith(".json"):
                df = pd.read_json(file)
                massprediction(df)