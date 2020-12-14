import models.en_2_fr as model
import models.predict as predict

#model.run_model()
predict.predict(model.input_word_index, model.output_word_index, model.max_input_length, "agenda")





