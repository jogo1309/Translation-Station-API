import numpy as np

from .read_data import pre_process_data

def data_formatting(input_file_path, outpur_file_path):
    #english dataset
    input_sentances, input_words, num_encoder_tokens, max_input_length = pre_process_data(input_file_path, False)

    #print(input_words)
    #print(num_input_tokens)
    #print(max_input_length)

    #french dataset
    output_sentances, output_words, num_decoder_tokens, max_output_length = pre_process_data(outpur_file_path, True)

    #print(output_words)
    #print(max_output_length)
    #print(max_output_length)

    #word dictionaries start at index 1 as 0 is the blank number
    input_word_index = dict([(word, i+1) for i, word in enumerate(input_words)])
    output_word_index = dict([(word , i+1) for i,word in enumerate(output_words)])
    #print(output_word_index)
   

    #make vectors 2 dims 1: amount of sentancrs 2: max length of sentance
    #map each sentance to it's integer equivilent

    #empty vectors of correct length
    encoder_input_vectors = np.zeros((len(input_sentances), max_input_length), dtype='float32')
    decoder_input_vectors = np.zeros((len(output_sentances), max_output_length), dtype='float32')
    decoder_output_vectors = np.zeros((len(output_sentances), max_output_length), dtype='float32')

    #create word - index sequence

    for i, (input_sentances, output_sentances) in enumerate(zip(input_sentances, output_sentances)):

        for j, word in enumerate(input_sentances.split()):
            #set word j in sentance i to the index in the dictionary
            encoder_input_vectors[i, j] = input_word_index[word]

        for j, word in enumerate(output_sentances.split()):
            decoder_input_vectors[i, j] = output_word_index[word]

            if(j > 0):
                decoder_output_vectors[i, j-1] = output_word_index[word]
    
    #print(decoder_output_vectors)

    #one hot encoding array's
    #each array has 3 dimentions 1: amount of sentances, 2: max length of a sentance, 3: amount of unique words in the dataset
    """

    #empty vectors of correct length
    encoder_input_vectors = np.zeros((len(input_sentances), max_input_length, num_encoder_tokens), dtype='float32')
    decoder_input_vectors = np.zeros((len(output_sentances), max_output_length, num_decoder_tokens), dtype='float32')
    decoder_output_vectors = np.zeros((len(output_sentances), max_output_length, num_decoder_tokens), dtype='float32')

    #create one hot encoding

    for i, (input_sentances, output_sentances) in enumerate(zip(input_sentances, output_sentances)):

        for j, word in enumerate(input_sentances.split()):

            #set the word j in sentance i to 1 where array is the index of the word in the dictionary 
            # e.g. if word 1 of sentance 1 is "a" set [0,0,2] to 1 (a is the third word in the dictionary)
            encoder_input_vectors[i, j, input_word_index[word]] = 1

        for j, word in enumerate(output_sentances.split()):
            
            decoder_input_vectors[i, j, output_word_index[word]] = 1
            
            if(j > 0):
                #decoder_output should not include the start [[ token and be one timestep ahead
             
                decoder_output_vectors[i, j, output_word_index[word]] = 1
    """

    return encoder_input_vectors, decoder_input_vectors, decoder_output_vectors, num_encoder_tokens, num_decoder_tokens, input_word_index, output_word_index, max_input_length

