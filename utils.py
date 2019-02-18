import numpy as np
import pandas as pd
import csv

def build_data_array(path, tag):
    train_data = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for line in reader:
            train_data.append(line)

    del (train_data[0])  # 删除.csv文件首行无关信息

    array = np.zeros(shape=(50000, 28))
    score = np.zeros(shape=(50000,))

    if tag == 'test':
        for row_idx in range(50000):
            for col_idx in range(1, 29):
                array[row_idx][col_idx - 1] = float(train_data[row_idx][col_idx])
        return array

    if tag == 'train':
        for row_idx in range(50000):
            for col_idx in range(1, 29):
                array[row_idx][col_idx - 1] = float(train_data[row_idx][col_idx])
            score[row_idx] = float(train_data[row_idx][29])
        return array, score
    else:
        raise()

def write_SubmitionFile(result, id_path):
    score = result
    with open(id_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        id = [row[0] for row in reader]
    del id[0]

    dataframe = pd.DataFrame({'id':id, 'score':score})
    dataframe.to_csv('data/submit.csv',index=False, sep=',', encoding='utf-8')
    print('Submitfile had saved...')
