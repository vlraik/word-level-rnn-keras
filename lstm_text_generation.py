from __future__ import print_function

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file

import numpy as np
import random
import sys
import os

#path = get_file('rj.txt')
path = "qoutes.txt"

try: 
    text = open(path).read().lower()
except UnicodeDecodeError:
    import codecs
    text = codecs.open(path, encoding='utf-8').read().lower()

print('corpus length:', len(text))

chars = set(text)
words = set(open('qoutes.txt').read().lower().split())

#words.update(' ')
#print("words",words[1:100])
'''for i in words:
    print("words",i)'''
#print(text)
print("chars:",type(chars))
print("words",type(words))
print("total number of unique words",len(words))
print("total number of unique chars", len(chars))
#print("chars:", chars)
#print("words:", words)

print('total words:', len(words))
word_indices = dict((c, i) for i, c in enumerate(words)) #change it to list_words perhaps
indices_word = dict((i, c) for i, c in enumerate(words))

print("word_indices", type(word_indices), "length:",len(word_indices) )
print("indices_words", type(indices_word), "length", len(indices_word))
#DONT EDIT ABOVE THIS


#print(word_indices['\n'])

# cut the text in semi-redundant sequences of maxlen words
maxlen = 30
step = 3
print("maxlen:",maxlen,"step:", step)
sentences = []
next_words = []
next_words= []
sentences1 = []
list_words = []
sentences1=text.lower().split('%%') #this is not being used, for now
#sentences1=text.lower().split()
sentences2=[]
#list_words=str(sentences1).split()
list_words=text.lower().split()

print("sentences1:",type(sentences1), "list_words", type(list_words))
#print("sentences1:", sentences1[0:10])
#print("list_words:", list_words[0:10])


#print(type(sentences2))
#print(list_words[0])
#list_words[0] = 'society'
#list_words[-2] = list_words[-1]
#print(sentences1[0:10])


for i in range(0,len(list_words)-maxlen, step):
    sentences2 = ' '.join(list_words[i: i + maxlen])
    #print("sentences",i,sentences2)
    sentences.append(sentences2)
    #print(i, sentences)
    next_words.append((list_words[i + maxlen]))
    #next_words.append(list_words[i+maxlen+1])
    #print(i,next_words)
print('nb sequences(length of sentences):', len(sentences))
print("length of next_word",len(next_words))
#print(sentences)



#print("sentences:", sentences[0:10])
#print("sentences2:", sentences2[:])


#DONT EDIT CODE BELOW THIS POINT, IT WORKS.

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(words)), dtype=np.bool)
y = np.zeros((len(sentences), len(words)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, word in enumerate(sentence.split()):
        #print(i,t,word)
        X[i, t, word_indices[word]] = 1
    y[i, word_indices[next_words[i]]] = 1


# build the model: 2 stacked LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(words))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(words)))
#model.add(Dense(1000))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

if os.path.isfile('weights'):
    model.load_weights('weights')

def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 50):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    #model.fit(X, y, batch_size=64, nb_epoch=2)
    model.save_weights('weights',overwrite=True)

    start_index = random.randint(0, len(list_words) - maxlen - 1)

    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        #lengthofsentence = len(list_words[start_index:start_index + maxlen])
        #sentence = text[start_index: start_index + maxlen]
        sentence = list_words[start_index: start_index + maxlen]
        #print(lengthofsentence)
        #sentence = text[start_index: lengthofsentence]
        generated += ' '.join(sentence)
        #generated += sentence
        print('----- Generating with seed: "' , sentence , '"')
        print()
        sys.stdout.write(generated)
        print()


        for i in range(30):
            x = np.zeros((1, maxlen, len(words)))
            for t, word in enumerate(sentence):
                print(t,word)
                x[0, t, word_indices[word]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_word = indices_word[next_index]
            generated += next_word
            print(generated)
            print('hi', sentence)
            sentence = sentence[1:].extend(next_word)
            #sentence = sentence[1:] + next_word
            print(sentence)
            sys.stdout.write(next_word)
            sys.stdout.flush()
        print()
#model.save_weights('weights')