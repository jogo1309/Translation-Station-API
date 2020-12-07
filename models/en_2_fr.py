
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from .format_data import data_formatting

import numpy as np

#get the vectors for the model

encoder_input_vector, decoder_input_vector, decoder_output_vector, num_encoder_tokens, num_decoder_tokens, input_word_index, output_word_index = data_formatting(
    Path(__file__).parent  / 'test data'  / 'europarl-v7-EN.txt', 
    Path(__file__).parent  / 'test data'  / 'europarl-v7-FR.txt')

#print(encoder_input_vector[1,12])

batch_size = 32
epochs = 50
latent_dim = 256


#set up encoder
encoder_inputs = keras.Input(shape=(None,))
encoder_embed =  layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_lstm = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_embed)
encoder_states = [state_h, state_c]

#set up decoder
decoder_inputs = keras.Input(shape=(None,))
decoder_embed = layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _  = decoder_lstm(decoder_embed, initial_state=encoder_states)
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs= decoder_dense(decoder_outputs)


"""
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None,))
x = layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
x, state_h, state_c = layers.LSTM(latent_dim,
                           return_state=True)(x)
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None,))
x = layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
x = layers.LSTM(latent_dim, return_sequences=True)(x, initial_state=encoder_states)
decoder_outputs = layers.Dense(num_decoder_tokens, activation='softmax')(x)
"""

#build and train the model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
#model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
print(num_encoder_tokens)
print("fitting model:")
model.fit([encoder_input_vector, decoder_input_vector], decoder_output_vector, batch_size=batch_size, epochs=epochs, validation_split=0.2)

model.save("en2fr")

#inference - used to predict sentances
#used to encode the sentance to be predicted
model = keras.models.load_model("en2fr")
#print(model.layers)

encoder_inputs = model.input[0]
encoder_outputs, encoder_state_h, encoder_state_c = model.layers[4].output
encoder_states = [encoder_state_h, encoder_state_c]
encoder_model = keras.Model(encoder_inputs, encoder_states)

#decoder setup
decoder_state_h = keras.Input(shape=(latent_dim,))
decoder_state_c = keras.Input(shape=(latent_dim,))
decoder_state = [decoder_state_h, decoder_state_c]

#embeddings of decoder seq
decoder_embed = model.layers[3]
inf_decoder_embed = decoder_embed(decoder_inputs)

#to predict set the first states to the states from the previous word
inf_decoder_outputs , inf_state_h, inf_state_c = decoder_lstm(inf_decoder_embed, initial_state=decoder_state)
inf_decoder_states = [inf_state_h, inf_state_c]

#pass output through dense layer
inf_decoder_outputs = decoder_dense(inf_decoder_outputs)

#decoder_model
inf_decoder_model = keras.Model([decoder_inputs] + decoder_state, [inf_decoder_outputs] + inf_decoder_states)

#inference function
def predict_sentance(eng_sentance):
    #encode input
    encoded_state = encoder_model.predict(eng_sentance)

    #generate empty french sentance
    frn_sentance = np.zeros((1,1))
    #set first word to start character [[
    frn_sentance[0,0] = output_word_index["[["]

    should_stop=False
    output_sentance = ''

    while not should_stop:
        output_tokens, h, c = inf_decoder_model.predict([frn_sentance] + encoded_state)
        found_word_index = np.argmax(output_tokens[0, -1, :])
        print(found_word_index)
        found_word = list(output_word_index.keys())[list(output_word_index.values()).index(found_word_index)]
        print(found_word)
        output_sentance += ' ' + found_word

        #exit condition
        if(found_word == ']]' or len(output_sentance) > 100):
            should_stop = True
        
        #update frn sentace to new state
        frn_sentance = np.zeros((1,1))
        frn_sentance[0,0] = found_word_index

        encoded_state = [h, c]
    return output_sentance
#test prediction
eng_test = "you have failed"
eng_vector = np.zeros((1,39))
print(input_word_index)
for j, word in enumerate(eng_test.split()):
    eng_vector[0, j] = input_word_index[word]
print(eng_vector)
decoded_sentance = predict_sentance(eng_vector)
print("english sentance: ", eng_test )
print("french sentace: " , decoded_sentance)
