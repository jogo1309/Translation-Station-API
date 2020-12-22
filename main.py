import models.en_2_fr as model
import models.predict as predict

from models.read_data import split_into_files
from pathlib import Path

#model.run_model()
predict.predict(model.input_word_index, model.output_word_index, model.max_input_length, "read this now")

#split_into_files(Path(__file__).parent / 'models' / 'Data'  / 'fra.txt')





