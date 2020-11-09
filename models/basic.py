import numpy as numpy
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path
import string

#get data line by line and process it
def pre_process_data(path, is_output):
    unique_sentances = []
    unique_words = set()
    f = open(path, encoding="utf8")
    for line in f.readlines():
        #convert to lower case, strip punctuation, remove newlines and add to sentance array
        line = line.lower().translate(str.maketrans('', '', string.punctuation)).replace("\n", "")
        if(is_output):
            #add start and end tokens to translations
            line = "[[" + line + "]]"
        unique_sentances.append(line)
        print(line)
        #add to unique words found
        words = line.split(" ")
        for word in words:
            #print(word)
            if word not in unique_words:
                unique_words.add(word)
    
    return unique_sentances, sorted(list(unique_words))
#english dataset
path = Path(__file__).parent  / 'test data'  / 'europarl-v7-EN.txt'
input_sentances, input_words = pre_process_data(path, False)
print(input_words)

#french dataset
path = Path(__file__).parent  / 'test data'  / 'europarl-v7-FR.txt'
output_sentances, output_words = pre_process_data(path, True)
print(output_words)

    


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