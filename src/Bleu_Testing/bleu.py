from nltk.translate.bleu_score import sentence_bleu
#reference = "il est très froid".split(" ")
#candidate = "il est trop lent".split(" ")
reference = [['regarde', 'par', 'ici']]
candidate = ['regarde', 'ici']
score = sentence_bleu(reference, candidate, weights=(0.5, 0))
print(score)