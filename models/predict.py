from tensorflow import keras
import numpy as np

def predict(input_word_index, output_word_index, max_length, sentance):
    latent_dim = 256


    #inference - used to predict sentances
    #used to encode the sentance to be predicted
    model_fr = keras.models.load_model("en2fr_10000")
    #print(model_fr.input)

    encoder_inputs = model_fr.input[0]
    encoder_outputs, encoder_state_h, encoder_state_c = model_fr.layers[4].output
    encoder_states = [encoder_state_h, encoder_state_c]
    #encoder_outputs, encoder_state_h = model_fr.layers[4].output
    #encoder_states = [encoder_state_h]
    encoder_model = keras.Model(encoder_inputs, encoder_states)

    #decoder setup
    decoder_state_h = keras.Input(shape=(latent_dim,), name="stateh")
    decoder_state_c = keras.Input(shape=(latent_dim,), name="statec")
    decoder_state = [decoder_state_h, decoder_state_c]
    #decoder_state = [decoder_state_h]

    #embeddings of decoder seq
    decoder_inputs = model_fr.input[1]
    decoder_embed = model_fr.layers[3]
    inf_decoder_embed = decoder_embed(decoder_inputs)

    #to predict set the first states to the states from the previous word7
    decoder_lstm = model_fr.layers[5]
    inf_decoder_outputs , inf_state_h, inf_state_c = decoder_lstm(inf_decoder_embed, initial_state=decoder_state)
    inf_decoder_states = [inf_state_h, inf_state_c]
    #decoder_gru = model_fr.layers[5]
    #inf_decoder_outputs , inf_state_h = decoder_gru(inf_decoder_embed, initial_state=decoder_state)
    #inf_decoder_states = [inf_state_h]

    #pass output through dense layer
    decoder_dense = model_fr.layers[6]
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
            #output_tokens, h = inf_decoder_model.predict([frn_sentance] + [encoded_state])
            found_word_index = np.argmax(output_tokens[0, -1, :])
            found_word = list(output_word_index.keys())[list(output_word_index.values()).index(found_word_index)]
            output_sentance += ' ' + found_word

            #exit condition
            if(found_word == ']]' or len(output_sentance) > 100):
                should_stop = True
            
            #update frn sentace to new state
            frn_sentance = np.zeros((1,1))
            frn_sentance[0,0] = found_word_index

            encoded_state = [h, c]
            #encoded_state = [h]
        return output_sentance.replace(']','')
    #test prediction
    print(max_length)
    eng_vector = np.zeros((1, max_length))
    print(input_word_index)
    for j, word in enumerate(sentance.split()):
        eng_vector[0, j] = input_word_index[word]
    print(eng_vector)
    decoded_sentance = predict_sentance(eng_vector)
    print("english sentance: ", sentance )
    print("french sentace: " , decoded_sentance)
