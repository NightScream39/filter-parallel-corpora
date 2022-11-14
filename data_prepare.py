import pandas as pd


def read_data_for_df():
    labels_df = pd.read_json(path_or_buf='LR model data/devset_100_2_labels.jsonl', lines=True)
    score_df = pd.read_json(path_or_buf='LR model data/WikiMatrix-scores.en-ru.json', lines=True)
    df = pd.read_csv('df.csv', sep="\t")
    return labels_df, score_df, df


def data_prepare_for_model():
    ces = []
    labels_df, scores_df, full_df = read_data_for_df()
    for score in scores_df.loc[:, 'CrossEntropyFilter']:
        ces.append(round(abs(score[0]-score[1]), 3))
    scores_df.loc[:, 'CES'] = pd.DataFrame(ces)

    # LenghtRateScore based on words
    lrs_word = []
    # LenghtRateScore based on chars
    lrs_char = []
    for ratio in scores_df.loc[:, 'LengthRatioFilter']:
        lrs_word.append(round(ratio['word'], 3))
        lrs_char.append(round(ratio['char'], 3))
    scores_df.loc[:, 'LRSword'] = pd.DataFrame(lrs_word)
    scores_df.loc[:, 'LRSchar'] = pd.DataFrame(lrs_char)

    # WordAligmentScore
    wa_score = []
    for score in scores_df.loc[:, 'WordAlignFilter']:
        wa_score.append(round((score[0] + score[1]), 3))
    scores_df.loc[:, 'WAscore'] = pd.DataFrame(wa_score)

    scores_df.loc[:, 'CosineSimScore'] = round(full_df.loc[:, 'Cosine similarity'], 3)
    scores_df.loc[:, 'Eng sent'] = full_df.loc[:, 'Eng sent']
    scores_df.loc[:, 'Ru sent'] = full_df.loc[:, 'Ru sent']
    scores_df.loc[:, 'Label'] = labels_df

    scores_df.drop(scores_df.columns[[0, 1, 2, 3, 4, 5]], axis=1, inplace=True)

    return scores_df


def prepare_csv_for_model():
    scores_df = data_prepare_for_model()
    scores_df.to_csv('df_for_model.csv', sep='\t')


if __name__ == '__main__':
    prepare_csv_for_model()



