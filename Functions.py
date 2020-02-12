import numpy as np 
from keras.preprocessing import sequence
from keras.datasets import imdb
word_idx = imdb.get_word_index()

def text_to_embeddedVector(arr):
    
    review_idx = np.array([word_idx.get(word, '#') for word in arr.split()])
    review_idx = np.delete(review_idx, np.where(review_idx == '#'))
    review_idx = review_idx.astype('int')
    review_idx += 3
    review_idx = review_idx.reshape(1, len(review_idx))
    review_padded = sequence.pad_sequences(review_idx, maxlen=500)
    
    return review_padded