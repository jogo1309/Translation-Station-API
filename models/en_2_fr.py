
#import tensorflow as tf
from numpy.core.fromnumeric import shape
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path

from .format_data import data_formatting

#get the vectors for the model

encoder_input_vector, decoder_input_vector, decoder_output_vector, num_encoder_tokens, num_decoder_tokens, = data_formatting(
    Path(__file__).parent  / 'test data'  / 'europarl-v7-EN.txt', 
    Path(__file__).parent  / 'test data'  / 'europarl-v7-FR.txt')

#print(encoder_input_vector[1,12])

batch_size = 64
epochs = 1
latent_dim = 256

#set up encoder
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
encoder_embed = layers.Embedding(num_encoder_tokens, latent_dim)(encoder_inputs)
encoder_lstm = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
encoder_states = [state_h, state_c]
"""
# Define an input sequence and process it.
encoder_inputs = keras.Input(shape=(None, num_encoder_tokens))
enc_emb =  layers.Embedding(num_encoder_tokens, latent_dim, mask_zero = True)(encoder_inputs)
encoder_lstm = layers.LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder_lstm(enc_emb)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]
"""

#set up decoder
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
decoder_embed = layers.Embedding(num_decoder_tokens, latent_dim)(decoder_inputs)
decoder_outputs, _, _  = layers.LSTM(latent_dim, return_sequences=True, return_state=True)(decoder_inputs, initial_state=encoder_states)

decoder_outputs= layers.Dense(num_decoder_tokens, activation='softmax')(decoder_outputs)
"""
# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = keras.Input(shape=(None, num_decoder_tokens))
dec_emb_layer = layers.Embedding(num_decoder_tokens, latent_dim, mask_zero = True)
dec_emb = dec_emb_layer(decoder_inputs)
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = layers.LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb,
                                     initial_state=encoder_states)
decoder_dense = layers.Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
"""
#build and train the model
model = keras.Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.summary()

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])
print(num_encoder_tokens)
print("fitting model:")
model.fit([encoder_input_vector, decoder_input_vector], decoder_output_vector, batch_size=64, epochs=100, validation_split=0.2)

