
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from .format_data import data_formatting

import numpy as np

#get the vectors for the model

encoder_input_vector, decoder_input_vector, decoder_output_vector, num_encoder_tokens, num_decoder_tokens, input_word_index, output_word_index, max_input_length = data_formatting(
    Path(__file__).parent  / 'test data'  / 'europarl-v7-EN-1000.txt', 
    Path(__file__).parent  / 'test data'  / 'europarl-v7-FR-1000.txt')

#print(encoder_input_vector[1,12])

def run_model():

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

    #build and train the model
    model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.summary()

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
    print(num_encoder_tokens)
    print("fitting model:")
    model.fit([encoder_input_vector, decoder_input_vector], decoder_output_vector, batch_size=batch_size, epochs=epochs, validation_split=0.2)

    model.save("en2fr")


