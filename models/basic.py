import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
import string

#get data line by line
path = Path(__file__).parent  / 'test data'  / 'europarl-v7-EN.txt'

f = open(path, encoding="utf8")

input_sentances = []
output_sentances = []

input_words = set()
output_words = set()


for line in f.readlines():
    #convert to lower case, strip punctuation, remove newlines and add to sentance array
    line = line.lower().translate(str.maketrans('', '', string.punctuation)).replace("\n", "")
    input_sentances.append(line)
    print(line)
    #add to unique words found
    words = line.split(" ")
    for word in words:
        #print(word)
        if word not in input_words:
            input_words.add(word)

print(sorted(list(input_words)))

    


"""
# Define Sequential model with 3 layers
model = keras.Sequential(
    [
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(4, name="layer3"),
    ]
)
# Call model on a test input
x = tf.ones((3, 3))
y = model(x)
print(model.layers) 

for dSet in dataset:
    text_vectorizer.adapt(dSet[0])
    dataset= dataset.map(
        lambda x, y: (text_vectorizer(x), y)
    )
for dSet in dataset:
    print(dSet[0])

"""