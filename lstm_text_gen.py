# -*- coding: utf-8 -*-
"""
Created on Sun Mar 26 15:25:53 2017

@author: JunTaniguchi
"""

import os
import sys
import glob
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.optimizer import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random

# 存在するtextファイル全てを取り込む
text_list = []
book_list = glob.glob("./book/*.txt") # Mac
for book_txt in book_list:
    with open(book_txt) as book_file:
        text = book_file.decode("shift_jis")
        text_list.append(text)

# i文字ずつIDを振る
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars)) # 文字 → ID
indices_char = dict((i, c) for i, c in enumerate(chars)) # ID → 文字


# テキストをmaxlen単位に区切って、その次に来る文字を予測する。
maxlen = 20
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])

# テキストのIDベクトル化
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer)


def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# 学習させて、テキストを生成する
for iteration in range(1, 60):
    print()
    print('-' * 50)
    print('繰り返し=', iteration)
    model.fit(X, y, batch_size=128, nb_epoch=1)
    # ランダムにテキストのシードを選ぶ
    start_index = random.randint(0, len(text) - maxlen - 1)
    # 多様性のパラメータごとに文を生成する
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('---多様性=', diversity)
        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('---シード="' + sentences + '"')
        sys.stdout.write(generated)
        # シードを元にテキストを自動で生成する
        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            # 次に来る文字を予測
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            # 既存の文章に予測した1文字を足す
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()