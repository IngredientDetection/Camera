from gensim.models import Word2Vec


from gensim.models import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import TfidfEmbeddingVectorizer
import MeanEmbeddingVectorizer
from collections import defaultdict
import pandas as pd

import numpy as np
import config
## 콘텐츠 기반 추천시스템

recipe_data = pd.read_csv("data\pre_tmdb_recipe3.csv",encoding='cp949')


#### 사용자가 특정 아이템을 선호하는 경우, 그 아이템과 '비슷한' 콘텐츠를 가진 다른 아이템을 추천해주는 방식

recipe_core=recipe_data
recipe_core=recipe_core.fillna("")
#get corpus with the documents sorted in alphabetical order

def get_and_sort_corpus(data):
    corpus_sorted = []
    for doc in data.요리재료내용.values:
        doc_list = []
        doc_list = doc.split(" ")
        doc_list.sort()
        corpus_sorted.append(doc_list)
        doc_list = []
    return corpus_sorted

# calculate average length of each document
def get_window(corpus):
    lengths = [len(doc) for doc in corpus]
    avg_len = float(sum(lengths)) / len(lengths)
    return round(avg_len)


if __name__ == "__main__":
     # load in data
     data = recipe_core
     # parse the ingredients for each recipe
     corpus = get_and_sort_corpus(data)
     print(f"Length of corpus: {len(corpus)}")
     # train and save CBOW Word2Vec model
     model_cbow = Word2Vec(
         corpus, sg=0, workers=0, window=get_window(corpus), min_count=1, vector_size=100
     )
     model_cbow.save('./model_cbow2.bin')
     print("Word2Vec model successfully trained")