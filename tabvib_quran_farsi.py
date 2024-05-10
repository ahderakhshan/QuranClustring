import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


data = pd.read_csv("./quran.csv")
data = data["processed matn"]
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data)
inverted_index = {}
terms = tfidf_vectorizer.get_feature_names_out()
with open("./terms.txt", "w", encoding="utf-8-sig") as term_file:
    for term in terms:
        term_file.write(term+"\n")
all_vectors = []
for val in tfidf_matrix:
    vector_of_text = np.zeros(6428)
    index = val.indices
    data = val.data
    for i in range(len(index)):
        vector_of_text[index[i]] = data[i]
    all_vectors.append(vector_of_text)
all_vectors = np.array(all_vectors)
np.save("./ayeh_vectors.npy", all_vectors)
