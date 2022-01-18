import re
import pickle
import sys
import string
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def process_tweet(tweet):
    result = re.sub("@\w+", '', tweet)
    result = re.sub('RT\s*:', '', result)
    result = re.sub('&lt;3', '', result)
    result = re.sub(':D+', '', result)
    result = re.sub("—", '', result)
    result = re.sub("–", '', result)
    result = re.sub("“", '', result)
    result = re.sub("…", '', result)
    result = re.sub('&gt;', '', result)
    result = result.translate(str.maketrans('', '', string.punctuation)).lower()
    result = " ".join([t for t in result.split() if not t.startswith('http')])
    return re.sub("\s+", " ", result.strip())

if __name__ == "__main__":
    model = pickle.load(open("best_model.pickle", "rb"))

    vectorizer = pickle.load(open("vectorizer.pickle", 'rb'))

    input_text = sys.argv[1]  
    vector_input = vectorizer.transform([process_tweet(input_text)])
    predicted_class = model.predict(np.array(vector_input.todense()))[0]

    if predicted_class == 0:
        print("Разговорный стиль текста")
    elif predicted_class == 1:
        print("Художественный стиль текста")
    elif predicted_class == 2:
        print("Технический стиль текста")
    else:
        assert False  
