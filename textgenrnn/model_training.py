import pickle

from keras.callbacks import LearningRateScheduler, Callback
from keras.models import Model, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import Sequence
from keras import backend as K
from .utils import textgenrnn_encode_cat
import numpy as np

from functools import lru_cache

def load_cache(dataset):
    try:
        with open(f'cache-{dataset}.pkl', 'rb') as f:
            cache = pickle.load(f)
            print("Loaded cache.")
            return cache
    except:
        return None

def save_cache(cache, dataset):
    print("Saving cache...")
    try:
        with open(f'cache-{dataset}.pkl', 'xb') as f:
            print("Saving cache ...")
            pickle.dump(cache, f)
    except FileExistsError:
        print("Cache exists.")

def generate_sequences_from_texts(texts, indices_list,
                                  textgenrnn, context_labels,
                                  batch_size=128, dataset='train'):
    is_words = textgenrnn.config['word_level']
    is_single = textgenrnn.config['single_text']
    max_length = textgenrnn.config['max_length']
    meta_token = textgenrnn.META_TOKEN

    cache = load_cache(dataset) or [None] * indices_list.shape[0]

    def get_xy(row):

        if cache[row] is not None:
            return cache[row]

        text_index = indices_list[row, 0]
        end_index = indices_list[row, 1]

        text = texts[text_index]

        if not is_single:
            text = [meta_token] + list(text) + [meta_token]

        if end_index > max_length:
            x = text[end_index - max_length: end_index + 1]
        else:
            x = text[0: end_index + 1]
        y = text[end_index + 1]
        if y in textgenrnn.vocab:
            x = process_sequence([x], textgenrnn, new_tokenizer)
            y = textgenrnn.vocab[y]
            ret = x, y
        else:
            ret = None, None

        cache[row] = ret
        return ret

    if is_words:
        new_tokenizer = Tokenizer(filters='', char_level=True)
        new_tokenizer.word_index = textgenrnn.vocab
    else:
        new_tokenizer = textgenrnn.tokenizer

    while True:

        X_batch = []
        Y_batch = []
        context_batch = []
        count_batch = 0

        for row in np.random.permutation(range(indices_list.shape[0])):

            x, y = get_xy(row)

            if x is None:
                continue

            X_batch.append(x)
            Y_batch.append(y)

            if context_labels is not None:
                context_batch.append(context_labels[text_index])

            count_batch += 1

            if count_batch % batch_size == 0:
                X_batch = np.squeeze(np.array(X_batch))

                # too big too cache
                #Y_batch = textgenrnn_encode_cat(Y_batch, textgenrnn.vocab)
                Y_batch = np.squeeze(np.array(Y_batch))

                context_batch = np.squeeze(np.array(context_batch))

                # print(X_batch.shape)

                if context_labels is not None:
                    yield ([X_batch, context_batch], [Y_batch, Y_batch])
                else:
                    yield (X_batch, Y_batch)
                X_batch = []
                Y_batch = []
                context_batch = []
                count_batch = 0

        save_cache(cache, dataset)


def process_sequence(X, textgenrnn, new_tokenizer):
    X = new_tokenizer.texts_to_sequences(X)
    X = sequence.pad_sequences(
        X, maxlen=textgenrnn.config['max_length'])

    return X
