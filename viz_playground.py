from utilities.utils import visualize_embeddings
from gensim.models import Word2Vec


model = Word2Vec.load("trained_models/dutch/nl_d300.model")
model.init_sims(replace=True)

# words = ["car", "train", "bike", "taxi", "cab", "dog",
#          "cat", "cow", "pig", "apple", "banana", "plane", "airplane",
#          "one", "two", "three", "four", "adult"]

words = ["aap", "hond", "kat", "koe", "auto", "fiets", "taxi", "vliegtuig", "trein",
         "een", "twee", "drie", "vier", "vijf", "zesThe Dutch "]

visualize_embeddings(model, words)