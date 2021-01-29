#import models.en_2_fr as model
#import models.predict as predict

#from models.read_data import split_into_files
#from pathlib import Path

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

from . import routes

#model.run_model("GRU")
#predict.predict("GRU", model.input_word_index, model.output_word_index, model.max_input_length, "you've failed")

#split_into_files(Path(__file__).parent / 'models' / 'Data'  / 'fra.txt')





