import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, classification_report
import data_prepare as data_prepare
import laser as laser
import matplotlib.pyplot as plt
import seaborn as sns



# Read csv to Pandas dataframe
def read_csv_to_dataframe():
    df = pd.read_csv('LR model data/df_for_model.csv', sep="\t")
    return df


# Preparing data for training model
def prepare_data_for_model_train():
    df_train = read_csv_to_dataframe()
    df_train = df_train.head(300)
    X = df_train[['CES', 'LRSword', 'LRSchar', 'WAscore', 'CosineSimScore']]
    y = df_train['Label']
    return df_train, X, y


def prepare_data_for_predict():
    # Preparing data for model predict
    df = read_csv_to_dataframe()
    x_full_for_test = df[['CES', 'LRSword', 'LRSchar', 'WAscore', 'CosineSimScore']]
    return x_full_for_test


def train_predict_model():
    # Split data into training and test
    X, y = prepare_data_for_model_train()[1:]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=41)
    # Set weigths for the classes (unbalance classes)
    w = {0: 7, 1: 1}
    # Training LogisticRegression model on labeled data with cross-validation
    n_folds = 10
    C_values = [0.001, 0.01, 0.05, 0.1, 1., 100.]
    lr_cv = LogisticRegressionCV(Cs=C_values, cv=n_folds, penalty='l2',
                           refit=True, scoring='recall',
                           solver='liblinear', random_state=40,
                           fit_intercept=False, class_weight=w).fit(X_train, y_train)
    # Predict labels (good or bad translation) for unlabeled data
    x_full = prepare_data_for_predict()
    y300_pred = lr_cv.predict(X_test)
    y_pred = lr_cv.predict(x_full[300:])
    # Union prepared labels with predicted labels
    df_train = prepare_data_for_model_train()[0]
    numpyarray = np.append(df_train['Label'], y_pred)
    return {'numpyarray': numpyarray}


def create_labeled_dataframes():
    # Create dataframes for good pair sents and for bad pairs.
    df = read_csv_to_dataframe()
    df.drop('Label', inplace=True, axis=1)
    numpyarray = train_predict_model()['numpyarray']
    df['Label'] = pd.DataFrame(numpyarray)
    df_negative_pairs = df.loc[df['Label'] == 0]
    df_positive_pairs = df.loc[df['Label'] != 0]
    return df_negative_pairs, df_positive_pairs


def save_sorted_dataframes():
    df_negative, df_positive = create_labeled_dataframes()
    # Saved good and bad pairs into different files
    with open('datasets/WikiMatrix-filteredDROPbyComplexFilter.en-ru.en', "w") as en:
        for sent in df_negative['Eng sent']:
            en.write(sent + '\n')

    with open('datasets/WikiMatrix-filteredDROPbyComplexFilter.en-ru.ru', "w") as ru:
        for sent in df_negative['Ru sent']:
            ru.write(sent + '\n')

    with open('datasets/WikiMatrix-filteredbyComplexFilter.en-ru.en', "w") as en:
        for sent in df_positive['Eng sent']:
            en.write(sent + '\n')

    with open('datasets/WikiMatrix-filteredbyComplexFilter.en-ru.ru', "w") as ru:
        for sent in df_positive['Ru sent']:
            ru.write(sent + '\n')


def filtered_parallel_corpora():
    # laser.cosine_similarity_scores()
    data_prepare.prepare_csv_for_model()
    save_sorted_dataframes()


if __name__ == '__main__':
    filtered_parallel_corpora()
