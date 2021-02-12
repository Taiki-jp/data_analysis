# ================================================ #
# *         Import Some Libraries
# ================================================ #

from my_setting import FindsDir, SetsPath
SetsPath().set()
import os, sys
import tensorflow as tf
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.keras.backend.set_floatx('float32')
from utils import PreProcess, Utils
from glob import glob
from pprint import pprint
import numpy as np
import pandas as pd
from tqdm import tqdm

# ================================================ #
# *          モデルの読み込みと作成
# ================================================ #

modelDirPath = "c:/users/taiki/sleep_study/models/"
# 後の方が時間的に新しい順になっている，w, r, nr1, nr2, nr34
modelList = glob(modelDirPath+'*')
print("*** this is model list ***")
pprint(modelList)
print("一番新しいモデルが最後に来ていることを確認")
m_findsDir = FindsDir("sleep")

# ================================================ #
# *                データの作成
# ================================================ #


#inputFileName = input("*** 被験者データを入れてください *** \n")
for loop_num, name in enumerate(Utils().name_list[::-1]):
    print("だれだれの実験をやっています", name)
    m_preProcess = PreProcess(project=m_findsDir.returnDirName(), input_file_name=name)
    (x_test, y_test) = m_preProcess.loadData(is_split=True)
    # x_test に関しては前処理が必要(Noneの処理，x_testの正規化)
    (x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
    m_preProcess.maxNorm(x_test)
    # データサイズを入れてこれを基にデータフレームを（行数, 睡眠段階数）作る
    datasize = y_test.shape[0]
    ss_num = 5
    df_label = pd.DataFrame(np.zeros(datasize*ss_num).reshape(datasize, ss_num))

    # TODO : データフレームを見やすくするためにラベルを用意
    labels_list = ["wake", "rem", "nr1", "nr2", "nr34"]
    list4read_models = np.arange(-1, -6, -1) - 5*loop_num
    for num, i in tqdm(enumerate(list4read_models)):
        model = tf.keras.models.load_model(modelList[i])
        pr = tf.math.softmax(model.predict(x_test))
        for k in range(datasize):
            try:
                df_label[num][k] = True if pr[k, 1].numpy() > pr[k, 0].numpy() else False
            except:
                print(k, num, i)
                sys.exit(1)

    # NNの自信のある部分の一致率
    nd_label = np.array(df_label)
    catched_rows = list()
    for row, _ in enumerate(range(datasize)):
        if nd_label[row].sum() == 1:
            catched_rows.append(row)

    # 一つしか２クラス分類がtrueといわなかったときの正解ラベルとの比較
    # 一つしかtrueといわなかったものの数
    one_true_patterns_num = len(catched_rows)
    one_true_patterns_correct = 0
    for row in catched_rows:
        #print("true", y_test[row])
        #print("predicted", nd_label[row])
        if nd_label[row][y_test[row]*(-1)]:
            one_true_patterns_correct += 1
    # 確率を表示
    print("NNが自信をもって判断したときの5段階一致率は", one_true_patterns_correct/one_true_patterns_num)
    # ちなみに何個そういうときがあるかを見る
    print("全体で何通りそのようなデータがあったか", one_true_patterns_num)
    #trueの数の確認
    wake_true = df_label[0].sum()
    rem_true = df_label[1].sum()
    nr1_true = df_label[2].sum()
    nr2_true = df_label[3].sum()
    nr34_true = df_label[4].sum()
    print(wake_true, rem_true, nr1_true, nr2_true, nr34_true)
    
# ================================================ #
# *          競合したときの処理方法
# ================================================ #



m_findsDir = FindsDir("sleep")
#inputFileName = input("*** 被験者データを入れてください *** \n")
for loop_num, name in enumerate(Utils().name_list[::-1]):
    #name = "H_Li"
    print("だれだれの実験をやっています", name)
    m_preProcess = PreProcess(project=m_findsDir.returnDirName(), input_file_name=name)
    (x_test, y_test) = m_preProcess.loadData(is_split=True)
    # x_test に関しては前処理が必要(Noneの処理，x_testの正規化)
    (x_test, y_test) = m_preProcess.catchNone(x_test, y_test)
    m_preProcess.maxNorm(x_test)
    # データサイズを入れてこれを基にデータフレームを（行数, 睡眠段階数）作る
    datasize = y_test.shape[0]
    ss_num = 5
    df_label = pd.DataFrame(np.zeros(datasize*ss_num).reshape(datasize, ss_num))

    # TODO : データフレームを見やすくするためにラベルを用意
    labels_list = ["wake", "rem", "nr1", "nr2", "nr34"]
    list4read_models = np.arange(-1, -6, -1) - 5*loop_num
    for num, i in tqdm(enumerate(list4read_models)):
        model = tf.keras.models.load_model(modelList[i])
        pr = tf.math.softmax(model.predict(x_test))
        for k in range(datasize):
            try:
                df_label[num][k] = True if pr[k, 1].numpy() > pr[k, 0].numpy() else False
            except:
                print(k, num, i)
                sys.exit(1)
    # 複数のモデルを作成する（複数のラベルから最大のコンフィデンスのものを選択するため）
    model4wake = tf.keras.models.load_model(modelList[list4read_models[0]])
    model4rem = tf.keras.models.load_model(modelList[list4read_models[1]])
    model4nr1 = tf.keras.models.load_model(modelList[list4read_models[2]])
    model4nr2 = tf.keras.models.load_model(modelList[list4read_models[3]])
    model4nr34 = tf.keras.models.load_model(modelList[list4read_models[4]])
    # NNの自信のある部分の一致率
    nd_label = np.array(df_label)
    catched_rows = list()
    no_labels_selected_rows = list()
    multiple_labels_selected_rows = list()
    for row, _ in enumerate(range(datasize)):
        if nd_label[row].sum() == 1:
            catched_rows.append(row)
        elif nd_label[row].sum() == 0:
            no_labels_selected_rows.append(row)
        else:
            multiple_labels_selected_rows.append(row)
    def select_one_label_from_multiples(multi_list, df):
        df2nd = np.array(df)
        # 複数のラベルが選ばれているとき選ばれた(true)のラベルに対して確率を計算して最大のものを選択する
        tmp = list()
        for data_num in multi_list:
            # ラベルがtrueのところの要素を返す(0:w,..., 4:nr34)
            check_labels = [i for i in range(5) if df2nd[data_num, i] == True]
            for label in check_labels:
                pr_w = 0
                pr_r = 0
                pr_n1 = 0
                pr_n2 = 0
                pr_n34 = 0
                if label == 0:
                    pr_w = tf.math.softmax(model4wake.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 1:
                    pr_r = tf.math.softmax(model4rem.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 2:
                    pr_n1 = tf.math.softmax(model4nr1.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 3:
                    pr_n2 = tf.math.softmax(model4nr2.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                elif label == 4:
                    pr_n34 = tf.math.softmax(model4nr34.predict(tf.expand_dims(x_test[data_num], axis=0)))[0][1].numpy()
                else:
                    print("error handle")
                    sys.exit(1)
            pr_list = np.array([pr_w, pr_r, pr_n1, pr_n2, pr_n34])
            # 最大の要素のインデックスを返す
            max_label = np.argmax(pr_list)
            tmp.append(max_label)
        return tmp
    selected_labels_in_multiples = select_one_label_from_multiples(multi_list=multiple_labels_selected_rows, df=df_label)
    
    # 複数ラベルの正解ラベルとの比較
    # NOTE : 確率が0になる
    for i, row in enumerate(multiple_labels_selected_rows):
        one_true_patterns_correct = 0
        if selected_labels_in_multiples[i] == y_test[row]:
            one_true_patterns_correct += 1
    one_true_patterns_num = len(multiple_labels_selected_rows)
    print("NNが複数をもって判断したときの5段階一致率は", one_true_patterns_correct/one_true_patterns_num)


    # 一つしか２クラス分類がtrueといわなかったときの正解ラベルとの比較
    # 一つしかtrueといわなかったものの数
    one_true_patterns_num = len(catched_rows)
    one_true_patterns_correct = 0
    for row in catched_rows:
        #print("true", y_test[row])
        #print("predicted", nd_label[row])
        if nd_label[row][y_test[row]*(-1)]:
            one_true_patterns_correct += 1
    # 確率を表示
    print("NNが自信をもって判断したときの5段階一致率は", one_true_patterns_correct/one_true_patterns_num)
    # ちなみに何個そういうときがあるかを見る
    print("全体で何通りそのようなデータがあったか", one_true_patterns_num)
    #trueの数の確認
    wake_true = df_label[0].sum()
    rem_true = df_label[1].sum()
    nr1_true = df_label[2].sum()
    nr2_true = df_label[3].sum()
    nr34_true = df_label[4].sum()
    print(wake_true, rem_true, nr1_true, nr2_true, nr34_true)   

"""
# テストデータ全体の数の確認
for name in Utils().name_list:
    print(f"{name}を読み込んでいます")
    m_preProcess = PreProcess(project=m_findsDir.returnDirName(), input_file_name=name)
    (x_test, y_test) = m_preProcess.loadData(is_split=True)
    print(y_test.shape)

# 外部書き出し
df_label.to_csv(os.path.join(os.environ["SLEEP"], "datas", f"{name}"+".csv"))
"""