
#import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from .format_data import data_formatting

import numpy as np

#get the vectors for the model

encoder_input_vector, decoder_input_vector, decoder_output_vector, num_encoder_tokens, num_decoder_tokens, input_word_index, output_word_index, max_input_length = data_formatting(
    Path(__file__).parent  / 'test data'  / 'engFile.txt', 
    Path(__file__).parent  / 'test data'  / 'frFile.txt')

#print(encoder_input_vector[1,12])

def LSTM_model(latent_dim):
    #set up encoder
    encoder_inputs = keras.Input(shape=(None,))
    encoder_embed =  layers.Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)

    encoder_lstm = layers.LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embed)
    encoder_states = [state_h, state_c]

    #set up decoder
    decoder_inputs = keras.Input(shape=(None,))
    decoder_embed = layers.Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _  = decoder_lstm(decoder_embed, initial_state=encoder_states)
    decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs= decoder_dense(decoder_outputs)

    return encoder_inputs, decoder_inputs, decoder_outputs

def BiDi_model(latent_dim):

    #set up encoder
    encoder_inputs = keras.Input(shape=(None,))
    encoder_embed =  layers.Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)   

    encoder_lstm = layers.Bidirectional(layers.LSTM(latent_dim, return_state=True))
    encoder_outputs, f_state_h, f_state_c, b_state_h, b_state_c = encoder_lstm(encoder_embed)
    state_h = keras.layers.Concatenate()([f_state_h, b_state_h])
    state_c = keras.layers.Concatenate()([f_state_c, b_state_c])
    encoder_states = [state_h, state_c] 

    #decoder
    decoder_inputs = keras.Input(shape=(None,))
    decoder_embed = layers.Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_lstm = layers.LSTM(latent_dim*2, return_sequences=True, return_state=True)
    decoder_outputs, _, _  = decoder_lstm(decoder_embed, initial_state=encoder_states)
    decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs= decoder_dense(decoder_outputs)

    return encoder_inputs, decoder_inputs, decoder_outputs

def GRU_model(latent_dim):
    #set up encoder
    encoder_inputs = keras.Input(shape=(None,))
    encoder_embed =  layers.Embedding(num_encoder_tokens, latent_dim, mask_zero=True)(encoder_inputs)
    encoder_gru = layers.GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder_gru(encoder_embed)

    #decoder
    decoder_inputs = keras.Input(shape=(None,))
    decoder_embed = layers.Embedding(num_decoder_tokens, latent_dim, mask_zero=True)(decoder_inputs)
    decoder_gru = layers.GRU(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _ = decoder_gru(decoder_embed, initial_state=state_h)
    decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs= decoder_dense(decoder_outputs)

    return encoder_inputs, decoder_inputs, decoder_outputs

def run_model(model_type):

    batch_size = 128
    epochs = 50
    latent_dim = 256

    encoder_inputs = None
    decoder_inputs = None
    decoder_outputs = None

    #get right model definition
    if(model_type == "LSTM"):
        encoder_inputs, decoder_inputs, decoder_outputs = LSTM_model(latent_dim)
    elif(model_type == "BiDi"):
        encoder_inputs, decoder_inputs, decoder_outputs = BiDi_model(latent_dim)
    elif(model_type == "GRU"):
        encoder_inputs, decoder_inputs, decoder_outputs = GRU_model(latent_dim)
    else:
        print("ERROR: invalid model defined")

    if(encoder_inputs != None):

        #build and train the model
        model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
        model.summary()

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['acc'])
        #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=[keras.metrics.CategoricalAccuracy()])
        print(num_encoder_tokens)
        print("fitting model:")
        model.fit([encoder_input_vector, decoder_input_vector], decoder_output_vector, batch_size=batch_size, epochs=epochs, validation_split=0.2)

        model.save("en2fr")


