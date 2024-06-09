#1. Book recommendation on similar-user basis.:- Similar Books others Liked--> User-based recommendation (pandas-based-transformation-of data)
#2. Book recommendation on what we read.:- More Books Like These --> Content based recommendation (cosine-similarity)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OrdinalEncoder
import pandas as pd
import streamlit as st

img,title = st.columns([0.15,0.85])
img.image("logo-1.jpg")
title.header("Restaurant Books Recommendations")

def getbook():
    global books_df
    books_df = pd.read_csv("book1-100k.csv")
    books_df = books_df[['Id','Name','PublishYear','Language','Authors','Rating','ISBN']]

def trainmodel():
    global rating_df
    rating_df = pd.read_csv("user_rating_0_to_1000.csv")
    rating_df = rating_df[rating_df.Name!='Rating']
    ord = OrdinalEncoder()
    rating_df[['Rating','BNo']] = ord.fit_transform(rating_df[['Rating','Name']])
    global df
    df = rating_df.pivot_table(columns='BNo',index='ID',values='Rating')
    df.fillna(-1,inplace=True)
    global model_nn
    model_nn = NearestNeighbors(metric='cosine')
    model_nn.fit(df)

def getsimilarbooks(uid):
    dist, idx = model_nn.kneighbors(df[df.index == uid],n_neighbors=10)
    books2recommend_index= []
    tu = df.iloc[0]
    for x in idx[0]:
        su = df.iloc[x]
        for t,s,i in zip(tu,su,tu.index):
            if t==-1.0 and s!=-1.0:
                print(i,"--",t,s)
                books2recommend_index.append(i)
    books2recommend = rating_df[rating_df.BNo.isin(books2recommend_index)].Name.value_counts()
    global final_books2recommend
    final_books2recommend = books2recommend.head(10).index

trainmodel()

userid = st.text_input("Enter userid:")
if st.button("Recommend"):
    getsimilarbooks(int(userid))
    st.markdown(""":books:
                **Books liked by others**
                :books:""")
    st.table(final_books2recommend)
