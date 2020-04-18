from utilities.utils import visualize_embeddings
from gensim.models import Word2Vec

model = Word2Vec.load("trained_models/english/en_d100_st.model")
model.init_sims(replace=True)

words = ["monkey", "dog", "cat", "cow", "car", "bike", "taxi", "cab", "airplane", "plane", "train",
         "one", "two", "three", "four", "five", "six",
         "amsterdam", "london", "berlin", "rotterdam", "amsterdam",
         "netherlands", "germany", "england"]

# words = ["aap", "hond", "kat", "koe", "auto", "fiets", "taxi", "vliegtuig", "trein",
#          "een", "twee", "drie", "vier", "vijf", "zes", ".", "<SOS>", "<EOS>",
#          "amsterdam", "london", "berlin", "rotterdam", "manchester"]

visualize_embeddings(model, words)