import string
from src.errors.APIerror import APIError
from .main import app


import src.models.en_2_fr as model
import src.models.predict as predict

#from models.read_data import split_into_files
from pathlib import Path

MODEL_TYPES = ["BiDi", "LSTM", "GRU"]

@app.route('/translate/<model_id>/<sentance>')
def doPrediction(model_id, sentance):
    cus_punctuation = string.punctuation.replace("'", "")
    sentance = sentance.lower().translate(str.maketrans('', '', cus_punctuation)).replace("\n", "").replace(u'\xa0', u' ').strip()

    if(not model_id in MODEL_TYPES):
        raise APIError("Error: invalid model specified", 400)
    
    prediction = predict.predict(model_id, model.input_word_index, model.output_word_index, model.max_input_length, sentance)
    return {
        "eng": sentance,
        "fr": prediction
    }

#model.run_model("GRU")
#predict.predict("GRU", model.input_word_index, model.output_word_index, model.max_input_length, "you've failed")

#split_into_files(Path(__file__).parent / 'models' / 'Data'  / 'fra.txt')
