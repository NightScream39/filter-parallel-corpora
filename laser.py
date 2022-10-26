import pandas as pd
import numpy as np
import re
from laserembeddings import Laser
import scipy.spatial.distance as ds


def parse_file_to_array():
    sentences_en = []
    sentences_ru = []
    with open('WikiMatrix-filtered.en-ru.en') as en:
        for line_en in en:
            reg_filter = re.sub("\n", '', line_en)
            sentences_en.append(reg_filter)

    with open('WikiMatrix-filtered.en-ru.ru') as ru:
        for line_ru in ru:
            reg_filter = re.sub("\n", '', line_ru)
            sentences_ru.append(reg_filter)
    return np.array(sentences_en), np.array(sentences_ru)


def array_to_df():
    np_sent_en, np_sent_ru = parse_file_to_array()
    df = pd.DataFrame(np_sent_en, columns=['Eng sent'])
    df['Ru sent'] = pd.DataFrame(np_sent_ru)
    return df


def cosine_similarity_en_ru_to_df():
    df = array_to_df()
    laser = Laser()
    cosine_sim_array = []

    for sent_en, sent_ru in zip(list(df['Eng sent'][:100]), list(df['Ru sent'][:100])):
        embeddings = laser.embed_sentences(
            [sent_en,
             sent_ru],
            lang=['en', 'ru'])
        distance = 1 - ds.cosine(embeddings[0], embeddings[1])
        cosine_sim_array.append(distance)
    cosine_nparray = np.array(cosine_sim_array)
    df['Cosine similarity'] = pd.DataFrame(cosine_nparray)
    return df


def save_dataframe():
    df = cosine_similarity_en_ru_to_df()
    df.to_csv('df.csv', sep='\t')


def laser_filtered_sentences():
    df2 = pd.read_csv('df.csv', sep="\t")
    df2 = df2[df2['Cosine similarity'] >= 0.825]
    df2 = df2.reset_index(drop=True)
    print(type(df2['Eng sent']))
    numpyarr_eng = df2['Eng sent'].to_numpy()
    numpyarr_ru = df2['Ru sent'].to_numpy()
    with open('WikiMatrix-filteredbyLASER.en-ru.en', "w") as en:
        for sent in numpyarr_eng:
            en.write(sent + '\n')

    with open('WikiMatrix-filteredbyLASER.en-ru.ru', "w") as ru:
        for sent in numpyarr_ru:
            ru.write(sent + '\n')


def laser_filtered_drop_sentences():
    df3 = pd.read_csv('df.csv', sep="\t")
    df3 = df3[df3['Cosine similarity'] <= 0.825]
    df3 = df3.reset_index(drop=True)
    print(type(df3['Eng sent']))
    numpyarray_en = df3['Eng sent'].to_numpy()
    numpyarray_ru = df3['Ru sent'].to_numpy()
    with open('WikiMatrix-filteredDROPbyLASER.en-ru.en', "w") as en:
        for sent in numpyarray_en:
            en.write(sent + '\n')

    with open('WikiMatrix-filteredDROPbyLASER.en-ru.ru', "w") as ru:
        for sent in numpyarray_ru:
            ru.write(sent + '\n')


def filter_corpora_by_laser():
    save_dataframe()
    laser_filtered_sentences()
    laser_filtered_drop_sentences()


if __name__ == '__main__':
    filter_corpora_by_laser()