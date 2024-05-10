from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv

quran = pd.read_csv("./quran.csv", encoding='utf-8-sig')
data = np.load("./ayeh_vectors.npy")
with open("./terms.txt", "r", encoding="utf-8-sig") as terms_file:
    terms = terms_file.readlines()
subject_file = open("./50_babes/subjects.txt", encoding="utf-8-sig", mode="w")
n_cluster = 50
kmeans = KMeans(n_clusters=n_cluster, random_state=42)
kmeans.fit_transform(data)
for i in range(n_cluster):
    file_i = open(f"./50_babes/{i}.csv", mode="w", newline='', encoding='utf-8-sig')
    writer = csv.writer(file_i)
    writer.writerow(["sureh", "ayeh", "matn", "processed matn"])
    index_i = np.where(kmeans.labels_ == i)[0]
    for index in index_i:
        data = [quran["sureh"][index], quran["ayeh"][index], quran["matn"][index], quran["processed matn"][index]]
        writer.writerow(data)
    file_i.close()
    centers_i = kmeans.cluster_centers_[i]
    top_indeces = np.argsort(centers_i)[-10:][::-1]
    low_indeces = np.argsort(centers_i)[:10]
    subject_file.write(f"{i+1}\n")
    subject = ""
    for top_index in top_indeces:
        subject += terms[top_index].strip() + ","
    subject_file.write(subject + "\n")
    low_subj = ""
    for low_index in low_indeces:
        low_subj += terms[low_index].strip() + ","
    subject_file.write(low_subj + "\n")
    subject_file.write("*"*10 + "\n")

a=10

