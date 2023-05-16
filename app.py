from flask import Flask, request
from flask_cors import CORS, cross_origin
import re
from transformers import pipeline
import json

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

#Initialize the following line somewhere in main so that it doesn't get called again and again 
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                    "]+", re.UNICODE)
    return re.sub(emoj, '', data)

def preprocess(text):
    text = ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",text).split())
    text = text.lower()
    return text  

def remove_stop_words(text):
    stop_words = ['viral', 'fyp', 'foryoupage', 'foryou', 'xyzbca']
    words = text.split()
    return ' '.join([i for i in words if i not in stop_words])

def classify(hashtag, desc):

    if desc.strip() == '':
        return False

    options = [hashtag, f"not {hashtag}"]
    hypothesis_template = "The topic of this short is {}."

    try:
        text = remove_emojis(desc)
    except:
        text = ""
    
    text = preprocess(text)
    text = remove_stop_words(text)

    if text.strip() == "":
        return False

    res = classifier(sequences=text, candidate_labels=options, hypothesis_template=hypothesis_template)
    labels = res['labels']
    scores = res['scores']
    ind = labels.index(hashtag)

    if scores[ind] > 0.90:
        return True
    else:
        return False

@app.route("/classify", methods=['POST'])
@cross_origin()
def classifer_api():
    data = json.loads(request.data)
    return json.dumps(classify(data['hashtag'], data['description']))