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

# ================================================ #
# *          モデルの読み込みと作成
# ================================================ #

modelDirPath = "c:/users/taiki/sleep_study/models/"
modelList = glob(modelDirPath+'*')
print("*** this is model list ***")
pprint(modelList)
print("一番新しいモデルが最後に来ていることを確認")
model = tf.keras.models.load_model(modelList[-2])
# 入力と出力を決める
new_input = model.input
new_output = model.get_layer('my_attention2d').output
new_model = tf.keras.Model(new_input, new_output)

# ================================================ #
# *              モデルのコンパイル
# ================================================ #

new_model.compile(loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer = tf.keras.optimizers.Adam(),
                  metrics = ["accuracy"])

# ================================================ #
# *                データの作成
# ================================================ #

m_findsDir = FindsDir("sleep")
inputFileName = input("*** 被験者データを入れてください *** \n")
m_preProcess = PreProcess(project=m_findsDir.returnDirName(), 
                          input_file_name=inputFileName)

(x_train, y_train) = m_preProcess.makeDataSet()
m_preProcess.maxNorm(x_train)
x_train, y_train = m_preProcess.catchNone(x_train, y_train)
y_train = m_preProcess.changeLabel(y_train)

# nr34:155, nr2: 395, nr1: 37, rem: 165, wake: 41

x_nr34 = list()
x_nr2 = list()
x_nr1 = list()
x_rem = list()
x_wake = list()

for num, ss in enumerate(y_train):
    if ss == 0:
        x_nr34.append(x_train[num])
    elif ss == 1:
        x_nr2.append(x_train[num])
    elif ss == 2:
        x_nr1.append(x_train[num])
    elif ss == 3:
        x_rem.append(x_train[num])
    elif ss == 4:
        x_wake.append(x_train[num])

x_nr34 = np.array(x_nr34)
x_nr2 = np.array(x_nr2)
x_nr1 = np.array(x_nr1)
x_rem = np.array(x_rem)
x_wake = np.array(x_wake)

attentionArray = []
confArray = []

convertedArray = [x_nr1, x_nr2, x_nr34, x_rem, x_wake]

for num, inputs in enumerate(convertedArray):
    attention = new_model.predict(inputs)
    if num == 0:
        labelNum = 2
    elif num == 1:
        labelNum = 1
    elif num == 2:
        labelNum = 0
    elif num == 3:
        labelNum = 3
    elif num == 4:
        labelNum = 4
    else:
        labelNum = None
    conf = tf.math.softmax(model.predict(inputs))[:, labelNum]
    attentionArray.append(attention)
    confArray.append(conf)

pathRoot = "c:/users/taiki/sleep_study/figures/"
savedDirList = ["nr1_attention_train/",
                "nr2_attention_train/",
                "nr34_attention_train/",
                "rem_attention_train/",
                "wake_attention_train/"]
savedDirList = [pathRoot + savedDir for savedDir in savedDirList]

for num, target in enumerate(attentionArray):
    m_preProcess.checkPath(savedDirList[num])
    m_preProcess.simpleImage(image_array = target,
                             row_image_array = convertedArray[num],
                             file_path = savedDirList[num],
                             x_label = "time",
                             y_label = "frequency",
                             title_array = confArray[num])
    