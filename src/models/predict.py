from src.errors.APIerror import APIError
from tensorflow import keras
import numpy as np
import logging

logger = logging.getLogger()

lstm_model = keras.models.load_model("en2fr_LSTM")
bidi_model = keras.models.load_model("en2fr_bidi")
gru_model = keras.models.load_model("en2fr_GRU")

def load_LSTM(latent_dim):
    #load saved LSTM model
    model_fr = lstm_model

    #create encoder model
    encoder_inputs = model_fr.input[0]
    encoder_outputs, encoder_state_h, encoder_state_c = model_fr.layers[4].output
    encoder_states = [encoder_state_h, encoder_state_c]

    encoder_model = keras.Model(encoder_inputs, encoder_states)

    #create deocder model
    decoder_state_h = keras.Input(shape=(latent_dim,), name="stateh")
    decoder_state_c = keras.Input(shape=(latent_dim,), name="statec")
    decoder_state = [decoder_state_h, decoder_state_c]

    #embeddings of decoder seq
    decoder_inputs = model_fr.input[1]
    decoder_embed = model_fr.layers[3]
    inf_decoder_embed = decoder_embed(decoder_inputs)

    #to predict set the first states to the states from the previous word7
    decoder_lstm = model_fr.layers[5]
    inf_decoder_outputs , inf_state_h, inf_state_c = decoder_lstm(inf_decoder_embed, initial_state=decoder_state)
    inf_decoder_states = [inf_state_h, inf_state_c]
    

    #pass output through dense layer
    decoder_dense = model_fr.layers[6]
    inf_decoder_outputs = decoder_dense(inf_decoder_outputs)

    #decoder model
    inf_decoder_model = keras.Model([decoder_inputs] + decoder_state, [inf_decoder_outputs] + inf_decoder_states)

    return encoder_model, inf_decoder_model


def load_BiDi(latent_dim):
    #load bidirectional model
    model_fr = bidi_model

    #encoder model
    encoder_inputs = model_fr.input[0]
    encoder_outputs, encoder_f_h, encoder_f_c, encoder_b_h, encoder_b_c = model_fr.layers[3].output
    encoder_state_h = keras.layers.Concatenate()([encoder_f_h, encoder_b_h])
    encoder_state_c = keras.layers.Concatenate()([encoder_f_c, encoder_b_c])
    encoder_states = [encoder_state_h, encoder_state_c]

    encoder_model = keras.Model(encoder_inputs, encoder_states)

    #decoder model
    decoder_state_h = keras.Input(shape=(latent_dim*2,), name="stateh")
    decoder_state_c = keras.Input(shape=(latent_dim*2,), name="statec")
    decoder_state = [decoder_state_h, decoder_state_c]

    #embeddings of decoder seq
    decoder_inputs = model_fr.input[1]
    decoder_embed = model_fr.layers[4]
    inf_decoder_embed = decoder_embed(decoder_inputs)

    #to predict set the first states to the states from the previous word7
    decoder_lstm = model_fr.layers[7]
    inf_decoder_outputs , inf_state_h, inf_state_c = decoder_lstm(inf_decoder_embed, initial_state=decoder_state)
    inf_decoder_states = [inf_state_h, inf_state_c]
    

    #pass output through dense layer
    decoder_dense = model_fr.layers[8]
    inf_decoder_outputs = decoder_dense(inf_decoder_outputs)

    #decoder_model
    inf_decoder_model = keras.Model([decoder_inputs] + decoder_state, [inf_decoder_outputs] + inf_decoder_states)

    return encoder_model, inf_decoder_model

def load_GRU(latent_dim):
    #load GRU model
    model_fr = gru_model

    #encoder model
    encoder_inputs = model_fr.input[0]
    encoder_outputs, encoder_state_h = model_fr.layers[4].output
    encoder_states = [encoder_state_h]

    encoder_model = keras.Model(encoder_inputs, encoder_states)

    #decoder model
    decoder_state_h = keras.Input(shape=(latent_dim,), name="stateh")
    decoder_state = [decoder_state_h]

    decoder_inputs = model_fr.input[1]
    decoder_embed = model_fr.layers[3]
    inf_decoder_embed = decoder_embed(decoder_inputs)

    decoder_gru = model_fr.layers[5]
    inf_decoder_outputs , inf_state_h = decoder_gru(inf_decoder_embed, initial_state=decoder_state)
    inf_decoder_states = [inf_state_h]

    decoder_dense = model_fr.layers[6]
    inf_decoder_outputs = decoder_dense(inf_decoder_outputs)

    #decoder_model
    inf_decoder_model = keras.Model([decoder_inputs] + decoder_state, [inf_decoder_outputs] + inf_decoder_states)

    return encoder_model, inf_decoder_model



def predict(model_type, input_word_index, output_word_index, max_length, sentance):
    try:
        latent_dim = 256

        #inference - used to predict sentances
    
        #get models so sentance can be encoded and decoded during prediction
        encoder_model = None
        inf_decoder_model = None
        if(model_type == "LSTM"):
            encoder_model, inf_decoder_model = load_LSTM(latent_dim)
        elif(model_type == "BiDi"):
            encoder_model, inf_decoder_model = load_BiDi(latent_dim)
        elif(model_type == "GRU"):
            encoder_model, inf_decoder_model = load_GRU(latent_dim)
        else:
            print("Error: Unkown model specified")

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
                if(model_type != "GRU"):
                    output_tokens, h, c = inf_decoder_model.predict([frn_sentance] + encoded_state)
                else:
                    output_tokens, h = inf_decoder_model.predict([frn_sentance] + [encoded_state])

                found_word_index = np.argmax(output_tokens[0, -1, :])
                found_word = list(output_word_index.keys())[list(output_word_index.values()).index(found_word_index)]
                output_sentance += ' ' + found_word

                #exit condition
                if(found_word == ']]' or len(output_sentance) > 100):
                    should_stop = True
                
                #update frn sentace to new state
                frn_sentance = np.zeros((1,1))
                frn_sentance[0,0] = found_word_index

                if(model_type != "GRU"):
                    encoded_state = [h, c]
                else:
                    encoded_state = [h]
                    
            return output_sentance.replace(']','')

        #test prediction
        if(encoder_model != None):
            #print(max_length)
            eng_vector = np.zeros((1, max_length))
            #print(input_word_index)
            try:
                for j, word in enumerate(sentance.split()):
                    eng_vector[0, j] = input_word_index[word]
            except Exception as e:
                raise APIError("KeyError: " + str(e) + " is not a known word", 400)

            #print(eng_vector)
            decoded_sentance = predict_sentance(eng_vector)
            print("english sentance: ", sentance )
            print("french sentace: " , decoded_sentance)
            return decoded_sentance
    except Exception as e:
        message = ""
        status_code = 500
        logger.exception("A Prediction Error Occured")
        if(hasattr(e, 'message')):
            message = e.message
        else:
            message = str(e)
        
        if(hasattr(e, 'status_code')):
            status_code = e.status_code
        
        raise APIError(message, status_code)
