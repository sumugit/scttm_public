""" 単語を訓練用とテスト用に分割 """

import pandas as pd
import numpy as np
import pickle
import sys
import random

# N_tdv = pd.read_pickle('../pickle_folder/q30/questions_bow_arr.pkl')
N_tdv = pd.read_pickle('../pickle_folder/paper/questions_bow_arr_paper.pkl')


def list_difference(list1, list2):
    result = list1.copy()
    for value in list2:
        if value in result:
            result.remove(value)
    
    return result

V_train = np.zeros(shape=(len(N_tdv), len(N_tdv[0]), len(N_tdv[0, 0])))
V_test = np.zeros(shape=(len(N_tdv), len(N_tdv[0]), len(N_tdv[0, 0])))

for t in range(len(N_tdv)):
    for d in range(len(N_tdv[t])):

        index_num = []
        for n, v in enumerate(N_tdv[t, d]):
            for i in range(int(v)):
                index_num.append(n)
        test_index = random.sample(index_num, int((len(index_num)*0.1)))
        train_index = list_difference(index_num, test_index)

        for num in test_index:
            V_test[t, d, num] += 1
        for num in train_index:
            V_train[t, d, num] += 1


# pd.to_pickle(V_train, '../pickle_folder/q30/questions_bow_arr_train.pkl')
# pd.to_pickle(V_test, '../pickle_folder/q30/questions_bow_arr_test.pkl')

pd.to_pickle(V_train, '../pickle_folder/paper/questions_bow_arr_paper_train.pkl')
pd.to_pickle(V_test, '../pickle_folder/paper/questions_bow_arr_paper_test.pkl')