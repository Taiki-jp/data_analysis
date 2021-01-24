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
from utils import PreProcess
from model_base import CreateModelBase
from glob import glob
from pprint import pprint

# ================================================ #
# *          モデルの読み込みと作成
# ================================================ #

modelDirPath = "c:/users/taiki/sleep_study/models/"
modelList = glob(modelDirPath+'*')
print("*** this is model list ***")
pprint(modelList)
print("一番新しいモデルが最後に来ていることを確認")
model = tf.keras.models.load_model(modelList[-1])
# 入力と出力を決める
new_input = model.input
new_output = model.get_layer('my_attention1d').output
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
inputFileName = input("*** 入力データを入れてください *** \n")
m_preProcess = PreProcess(project=m_findsDir.returnDirName(), 
                          input_file_name=inputFileName)

nr1_train, nr2_train, nr34_train, rem_train, wake_train = m_preProcess.makeEachSleepStageTrainData()
nr1_test, nr2_test, nr34_test, rem_test, wake_test = m_preProcess.makeEachSleepStageTrainData()

convertedArray = [nr1_train, 
                  nr1_test,
                  nr2_train,
                  nr2_test,
                  nr34_train,
                  nr34_test,
                  rem_train,
                  rem_test,
                  wake_train,
                  wake_test]

attentionArray = []
confArray = []

for num, inputs in enumerate(convertedArray):
    attention = new_model.predict(inputs)
    if num == 0 or num == 1:
        labelNum = 2
    elif num == 2 or num == 3:
        labelNum = 3
    elif num == 4 or num == 5:
        labelNum = 4
    elif num == 6 or num == 7:
        labelNum = 1
    else:
        labelNum = 0
    conf = tf.math.softmax(model.predict(inputs))[:, labelNum]
    attentionArray.append(attention)
    confArray.append(conf)

pathRoot = "c:/users/taiki/sleep_study/figures/"
savedDirList = ["nr1_attention_train/",
                "nr1_attention_test/",
                "nr2_attention_train/",
                "nr2_attention_test/",
                "nr34_attention_train/",
                "nr34_attention_test/",
                "rem_attention_train/",
                "rem_attention_test/",
                "wake_attention_train/",
                "wake_attention_test/"]
savedDirList = [pathRoot + savedDir for savedDir in savedDirList]

for num, target in enumerate(attentionArray):
    m_preProcess.checkPath(savedDirList[num])
    m_preProcess.simpleImage(image_array = target,
                             row_image_array = convertedArray[num],
                             file_path = savedDirList[num],
                             x_label = "time",
                             y_label = "freqency",
                             title_array = confArray[num])
    