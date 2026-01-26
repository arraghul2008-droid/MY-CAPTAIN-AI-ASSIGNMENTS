import ssl
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Activation
from tensorflow.keras.optimizers import RMSprop
import random
import sys

ssl._create_default_https_context = ssl._create_unverified_context

print("Loading text data...")

path = tf.keras.utils.get_file(
    'shakespeare.txt', 
    origin='https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)
text = open(path, 'rb').read().decode(encoding='utf-8').lower()

text = text[0:10000] 
print(f'Corpus length: {len(text)}')

chars = sorted(list(set(text)))
print(f'Total unique chars: {len(chars)}')
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i : i + maxlen])
    next_chars.append(text[i + maxlen])
print(f'Number of sequences: {len(sentences)}')

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

print('Building model...')
model = Sequential()
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

print("Training model (This will take a minute)...")
model.fit(x, y, batch_size=128, epochs=5)

def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

print("\n--- GENERATING TEXT ---")
start_index = random.randint(0, len(text) - maxlen - 1)
generated = ''
sentence = text[start_index : start_index + maxlen]
generated += sentence

print(f'Seed: "{sentence}"')
sys.stdout.write(generated)

for i in range(400): 
    x_pred = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x_pred[0, t, char_indices[char]] = 1.

    preds = model.predict(x_pred, verbose=0)[0]
    next_index = sample(preds, 0.5)
    next_char = indices_char[next_index]

    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()

print("\n\nDone!")