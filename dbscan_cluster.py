from sklearn.cluster import KMeans, DBSCAN, OPTICS
import numpy as np
import pandas as pd
import csv

quran = pd.read_csv("./quran.csv", encoding='utf-8-sig')
data = np.load("./ayeh_vectors.npy")
with open("./terms.txt", "r", encoding="utf-8-sig") as terms_file:
    terms = terms_file.readlines()
subject_file = open("./DBSCAN/subjects.txt", encoding="utf-8-sig", mode="w")
n_cluster = 50
#kmeans = KMeans(n_clusters=n_cluster, random_state=42)
kmeans = OPTICS()
kmeans.fit(data)
for i in np.unique(kmeans.labels_):
    file_i = open(f"./DBSCAN/{i}.csv", mode="w", newline='', encoding='utf-8-sig')
    writer = csv.writer(file_i)
    writer.writerow(["sureh", "ayeh", "matn", "processed matn"])
    index_i = np.where(kmeans.labels_ == i)[0]
    for index in index_i:
        data_write = [quran["sureh"][index], quran["ayeh"][index], quran["matn"][index], quran["processed matn"][index]]
        writer.writerow(data_write)
    file_i.close()
    centers_i = np.mean(data[kmeans.labels_ == i])
    top_indeces = np.argsort(centers_i)[-5:][::-1]
    subject = ""
    for top_index in top_indeces:
        subject += terms[top_index].strip() + ","
    subject_file.write(subject + "\n")

a=10

