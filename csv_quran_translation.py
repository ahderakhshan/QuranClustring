import pandas as pd
import csv
from hazm import Lemmatizer, Normalizer
from hazm.utils import stopwords_list
lemmatizer = Lemmatizer()
normalizer = Normalizer()

def preprocess_matn(matn):
    unallowed_chars = [".", "/", "!", "[", "]", ",", "،", ":", "«", "»"]
    for char in unallowed_chars:
        matn = matn.replace(char, "")
    matn = normalizer.normalize(matn)
    words = matn.split(" ")
    result = ""
    for word in words:
        if word not in stopwords_list():
            lemma = lemmatizer.lemmatize(word)
            if "#" not in lemma:
                result += lemma + " "
            else:
                lemma = lemma.split("#")[0]
                result += lemma + " "
    return result


quran_file_csv = open("./quran.csv", mode="w", newline='', encoding='utf-8-sig')
columns = ["sureh", "ayeh", "matn", "processed matn"]
quran_file_writer = csv.writer(quran_file_csv)
quran_file_writer.writerow(columns)

with open('./fa.gharaati.txt', 'r', encoding='utf-8-sig') as quran_file:
    for line in quran_file:
        line = line.strip()
        sureh_ayeh_matn = line.split("|")
        sureh, ayeh, matn = sureh_ayeh_matn[0], sureh_ayeh_matn[1], sureh_ayeh_matn[2]
        preprocessed_matn = preprocess_matn(matn)
        quran_file_writer.writerow([sureh, ayeh, matn, preprocessed_matn])


