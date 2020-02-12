from flask import Flask, render_template, request
from Functions import *
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
import tensorflow as tf
global graph,model

graph = tf.get_default_graph()


model2 = Sequential()
model2.add(Embedding(10000, 64))
model2.add(SimpleRNN(32))
model2.add(Dense(1, activation='sigmoid'))
model2.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

model2.load_weights('best_model_weights.h5')

app = Flask(__name__)

@app.route('/')
def home():
    print("Post is not running")
    return render_template("index.html")

@app.route('/', methods=['POST'])
def submit():
    if request.method == 'POST':

        print("Post is running")
        raw_review = request.form['review']
        review_padded = text_to_embeddedVector(raw_review)
        with graph.as_default():
            y = model2.predict(review_padded)
        y = y[0][0]*100
        y = np.int(np.round(np.interp(y, [0.0, 100.0], [1,10])))

        return render_template("index.html", results=y)

if __name__ == "__main__":
    app.run(debug=True)