import os
import sys
import logging
import unidecode
import ast

import numpy as np
import pandas as pd

from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from TfidfEmbeddingVectorizer import TfidfEmbeddingVectorizer
from MeanEmbeddingVectorizer import MeanEmbeddingVectorizer
import Word2Vec_train
from collections import defaultdict
import pandas as pd
import pickle
import csv
import numpy as np
import config
## 콘텐츠 기반 추천시스템

recipe_data = pd.read_csv("data\pre_tmdb_recipe3.csv",encoding='cp949')

recipe_data=recipe_data[['레시피일련번','food_name','요리방법별명','요리상황별명','요리재료별명','요리종류별명','요리재료내용','요리난이도명']]

#### 사용자가 특정 아이템을 선호하는 경우, 그 아이템과 '비슷한' 콘텐츠를 가진 다른 아이템을 추천해주는 방식

recipe_core=recipe_data[['레시피일련번','food_name','요리종류별명','요리재료내용']]
recipe_core=recipe_core.fillna("")
import config
#from ingredient_parser import ingredient_parser


def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.요리재료내용.values:
        doc_list = []
        doc_list = doc.split(" ")
        doc_list.sort()
        corpus_sorted.append(doc_list)
        doc_list = []
    return corpus_sorted


def get_recommendations(N, scores):
    """
    Top-N recomendations order by score
    """
    # load in recipe dataset
    df_recipes = recipe_core
    # order the scores with and filter to get the highest N scores
    top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:N]
    # create dataframe to load in recommendations
    recommendation = pd.DataFrame(columns=["음식명","요리재료내용","score"])
    count = 0
    for i in top:
        recommendation.at[count, "음식명"] = df_recipes["food_name"][i]
        recommendation.at[count, "요리재료내용"] =df_recipes["요리재료내용"][i]
        recommendation.at[count, "score"] = f"{scores[i]}"
        count += 1
    return recommendation



def get_recs(ingredients, N=5, mean=False):
    # load in word2vec model
    model = Word2Vec.load("./model_cbow2.bin")
    model.init_sims(replace=True)
    if model:
        print("Successfully loaded model")


    corpus = get_and_sort_corpus(recipe_core)

    if mean:
        # get average embdeddings for each document
        mean_vec_tr = MeanEmbeddingVectorizer(model)
        doc_vec = mean_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)
    else:
        # use TF-IDF as weights for each word embedding
        tfidf_vec_tr = TfidfEmbeddingVectorizer(model)
        tfidf_vec_tr.fit(corpus)
        doc_vec = tfidf_vec_tr.transform(corpus)
        doc_vec = [doc.reshape(1, -1) for doc in doc_vec]
        assert len(doc_vec) == len(corpus)

    # create embessing for input text
    input = ingredients
    # create tokens with elements
    input = input.split(",")

    if mean:
        input_embedding = mean_vec_tr.transform([input])[0].reshape(1, -1)
    else:
        input_embedding = tfidf_vec_tr.transform([input])[0].reshape(1, -1)

    # get cosine similarity between input embedding and all the document embeddings
    cos_sim = map(lambda x: cosine_similarity(input_embedding, x)[0][0], doc_vec)
    scores = list(cos_sim)
    # Filter top N recommendations
    recommendations = get_recommendations(N, scores)
    return recommendations


