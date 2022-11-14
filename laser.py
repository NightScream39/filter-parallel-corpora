import pandas as pd
import numpy as np
from laserembeddings import Laser
import scipy.spatial.distance as ds


def parse_file_to_array():
    sentences_en = []
    sentences_ru = []
    with open('datasets/WikiMatrix-filtered.en-ru.en') as en:
        patterns = ['\n', '"', '(', ')', '«', '»', '.']
        for line_en in en:
            for pattern in patterns:
                line_en = line_en.replace(pattern, '')
            sentences_en.append(line_en)

    with open('datasets/WikiMatrix-filtered.en-ru.ru') as ru:
        patterns = ['\n', '"', '(', ')', '«', '»', '.']
        for line_ru in ru:
            for pattern in patterns:
                line_ru = line_ru.replace(pattern, '')
            sentences_ru.append(line_ru)
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

    for sent_en, sent_ru in zip(list(df['Eng sent']), list(df['Ru sent'])):
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

#
# def laser_filtered_sentences():
#     df2 = pd.read_csv('df.csv', sep="\t")
#     df2 = df2[df2['Cosine similarity'] >= 0.825]
#     df2 = df2.reset_index(drop=True)
#     print(type(df2['Eng sent']))
#     numpyarr_eng = df2['Eng sent'].to_numpy()
#     numpyarr_ru = df2['Ru sent'].to_numpy()
#     with open('datasets/WikiMatrix-filteredbyLASER.en-ru.en', "w") as en:
#         for sent in numpyarr_eng:
#             en.write(sent + '\n')
#
#     with open('datasets/WikiMatrix-filteredbyLASER.en-ru.ru', "w") as ru:
#         for sent in numpyarr_ru:
#             ru.write(sent + '\n')
#
#
# def laser_filtered_drop_sentences():
#     df3 = pd.read_csv('df.csv', sep="\t")
#     df3 = df3[df3['Cosine similarity'] <= 0.825]
#     df3 = df3.reset_index(drop=True)
#     numpyarray_en = df3['Eng sent'].to_numpy()
#     numpyarray_ru = df3['Ru sent'].to_numpy()
#     with open('datasets/WikiMatrix-filteredDROPbyLASER.en-ru.en', "w") as en:
#         for sent in numpyarray_en:
#             en.write(sent + '\n')
#
#     with open('datasets/WikiMatrix-filteredDROPbyLASER.en-ru.ru', "w") as ru:
#         for sent in numpyarray_ru:
#             ru.write(sent + '\n')


def cosine_similarity_scores():
    print("Started counting cosine similarity for all Dataset... It may take a long time..")
    save_dataframe()
    # laser_filtered_sentences()
    # laser_filtered_drop_sentences()


if __name__ == '__main__':
    cosine_similarity_scores()

