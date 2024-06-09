import streamlit as st
import pandas as pd
from sklearn.neighbors import NearestNeighbors

img, head = st.columns([0.15,0.85])
img.image("logo-1.jpg")
head.header("Restaurant Movies Recommendation")

def trainmodel():
    global df,df1
    df = pd.read_csv("movies_collaborative.csv")
    df1 = df.pivot_table(columns="movieId",index="userId",values="rating")
    df1.fillna(0.0,inplace=True)
    global model_nn
    model_nn = NearestNeighbors(metric="cosine")
    model_nn.fit(df1)

def getsimilarmovies(id):
    uidx = id-1
    global model_nn
    dis, idx = model_nn.kneighbors([df1.iloc[uidx]])
    movies_target_user = df1.iloc[idx[uidx][0]]
    movies_similar_user = df1.iloc[idx[0][1]]
    movies_2_recommend_id = {}
    rm_id = []
    for id,t,s in zip(movies_target_user.index,movies_target_user,movies_similar_user):
        if t==0.0 and s!=0.0:
            #print(id,t,s)
            movies_2_recommend_id[id]=s
            rm_id.append(id)

    recommended_movies = df[df.movieId.isin(rm_id)].title.value_counts()
    global final_recommended_movies, frm_views
    final_recommended_movies = recommended_movies.head(10).index
    frm_views = recommended_movies.head(10).values

trainmodel()

uid = st.text_input("Enter your userId:")
if st.button("Submit"):
    getsimilarmovies(int(uid))
    st.markdown("************Recommended Movies to watch************")
    for i in range(0,len(final_recommended_movies),2):
        f,s = st.columns([0.5,0.5])
        fv,sv = st.columns([0.5,0.5])
        f.write(final_recommended_movies[i])
        fv.write("Views: "+str(frm_views[i]))
        s.write(final_recommended_movies[i+1])
        sv.write("Views: "+str(frm_views[i+1]))
        st.write("----------------------------------------------------------------------------------")
