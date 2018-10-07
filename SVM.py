import pandas as pd
import numpy as np
import pathlib
from tqdm import tqdm
from PIL import Image
from skimage import io, transform, color
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from sklearn import svm
from datetime import datetime
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split


path_train1 = sorted(
    list(pathlib.Path('full1_upload/training_data').resolve().glob('*.csv')))
path_test1 = sorted(
    list(pathlib.Path('full1_upload/test_feature').resolve().glob('*.csv')))
path_train2 = sorted(
    list(pathlib.Path('upload_full2/training_data').resolve().glob('*.csv')))
path_test2 = sorted(
    list(pathlib.Path('upload_full2/test_feature').resolve().glob('*.csv')))


def getdata():
    train_data_1 = []
    test_data_1 = []
    train_data_2 = []
    test_data_2 = []
    print('Load data')
    for i, j in enumerate(tqdm(path_train1)):
        train_data_1.append(pd.read_csv(str(j), low_memory=False))
    for i, j in enumerate(tqdm(path_test1)):
        test_data_1.append(pd.read_csv(str(j), low_memory=False))
    for i, j in enumerate(tqdm(path_train2)):
        train_data_2.append(pd.read_csv(str(j), low_memory=False))
    for i, j in enumerate(tqdm(path_test2)):
        test_data_2.append(pd.read_csv(str(j), low_memory=False))
    return train_data_1, test_data_1, train_data_2, test_data_2


def count(train=None, test=None):
    dataf_cnt = []
    dstaf_lbl = []
    test_cnt = []
    test_meta = []
    for x in tqdm(train):
        lbl = x['Label']
        x = x[x.columns[x.loc[0] == 'count']].fillna(0)
        lbl = lbl.iloc[2:]
        dataf_cnt.append(x.iloc[2:])
        dstaf_lbl.append(lbl.astype(bool))
    for x in tqdm(test):
        meta = x.iloc[:, 1:3]
        x = x[x.columns[x.loc[0] == 'count']].fillna(0)
        meta = meta.iloc[2:]
        meta.columns = ['machine', 'date']
        for i in meta.index:
            time_str = meta.at[i, 'date']
            dt = datetime.strptime(time_str, '%Y-%m-%d')
            time_str = dt.strftime("-%d/%m/%Y")
            meta.at[i, 'machine'] = meta.at[i, 'machine'] + time_str
        test_cnt.append(x.iloc[2:])
        test_meta.append(meta.iloc[:, :1])
    return dataf_cnt, dstaf_lbl, test_cnt, test_meta


def train(train_cnt, train_lbl, test_cnt, test_meta, output_name):
    acc_list = []
    for X, Y, C, A, B in tqdm(zip(train_cnt, train_lbl, test_cnt, test_meta, output_name)):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
        p_grid = [
            {'C': [0.01, 0.1, 1, 10, 100], 'gamma': ['auto', 0.1, 1, 10], 'kernel': ['sigmoid', 'rbf'],},
        ]
        clf_gscv = GridSearchCV(svm.SVR(), p_grid, verbose=2, return_train_score=True)
        clf_gscv.fit(X_train, Y_train)
        val_pred = clf_gscv.predict(X_test)
        rightans = 0
        for pred_lbl, true_lbl in tqdm(zip(val_pred, Y_test)):
            if (pred_lbl > 0.5 and true_lbl) or (pred_lbl < 0.5 and not true_lbl):
                rightans = rightans + 1
        val_accuracy = rightans / len(Y_test)
        acc_list.append(val_accuracy)
        pred = clf_gscv.predict(C)
        A.columns = ['id']
        A['Label'] = pred
        A.to_csv(B + 'results.csv', sep=',', index=False)
    return acc_list


if __name__ == '__main__':

    train1,test1,train2,test2 = getdata()
    train1_cnt, train1_lbl, test1_cnt, test1_meta = count(train1, test1)
    output_name1 = [fpath.name[:-7] + '1' for fpath in path_test1]
    predict1 = train(train1_cnt, train1_lbl,
                             test1_cnt, test1_meta, output_name1)
    train2_cnt, train2_lbl, test2_cnt, test2_meta = count(train2, test2)
    output_name2 = [fpath.name[:-7] + '2' for fpath in path_test2]
    predict2 = train(train2_cnt, train2_lbl,
                             test2_cnt, test2_meta, output_name2)
    print('Acc:')
    for acc, output_name in tqdm(zip(predict1, output_name1)):
        print(f'{output_name}: {acc}')
    for acc, output_name in tqdm(zip(predict2, output_name2)):
        print(f'{output_name}: {acc}')
