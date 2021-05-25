import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer 

from flask import Flask, render_template, url_for, request, jsonify      



app = Flask(__name__)

# Loading Vocabulary
with open(r"toxic_vectorized.pkl", "rb") as f:
    vocab_toxic = pickle.load(f)

with open(r"severe_toxic_vectorized.pkl", "rb") as f:
    vocab_severe = pickle.load(f)

with open(r"obscene_vectorized.pkl", "rb") as f:
    vocab_obscene = pickle.load(f)

with open(r"insult_vectorized.pkl", "rb") as f:
    vocab_insult = pickle.load(f)

with open(r"threat_vectorized.pkl", "rb") as f:
    vocab_threat = pickle.load(f)

with open(r"identity_hate_vectorized.pkl", "rb") as f:
    vocab_identity = pickle.load(f)

# Load the pickled RDF models
with open(r"lr_toxic_model.pkl", "rb") as f:
    model_toxic = pickle.load(f)

with open(r"lr_severe_toxic_model.pkl", "rb") as f:
    model_severe = pickle.load(f)

with open(r"lr_obscene_model.pkl", "rb") as f:
    model_obscene  = pickle.load(f)

with open(r"lr_insult_model.pkl", "rb") as f:
    model_insult  = pickle.load(f)

with open(r"lr_threat_model.pkl", "rb") as f:
    model_threat  = pickle.load(f)

with open(r"lr_identity_hate_model.pkl", "rb") as f:
    model_identity  = pickle.load(f)

# Render the HTML file for the home page
@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    
    # Take a string input from user
    input = request.form['text']
    data = [input]

    vectorizeData = vocab_toxic.transform(data)
    pred_toxic = model_toxic.predict_proba(vectorizeData)[:,1]

    vectorizeData = vocab_severe.transform(data)
    pred_severe = model_severe.predict_proba(vectorizeData)[:,1]

    vectorizeData = vocab_obscene.transform(data)
    pred_obscene = model_obscene.predict_proba(vectorizeData)[:,1]

    vectorizeData = vocab_threat.transform(data)
    pred_threat = model_threat.predict_proba(vectorizeData)[:,1]

    vectorizeData = vocab_insult.transform(data)
    pred_insult = model_insult.predict_proba(vectorizeData)[:,1]

    vectorizeData = vocab_identity.transform(data)
    pred_identity = model_identity.predict_proba(vectorizeData)[:,1]

    toxic = round(pred_toxic[0], 2)
    severe_toxic = round(pred_severe[0], 2)
    obscene = round(pred_obscene[0], 2)
    insult = round(pred_insult[0], 2)
    threat = round(pred_threat[0], 2)
    identity_hate = round(pred_identity[0], 2)




    return render_template('index.html', 
                            pred_toxic = 'Toxic: {}'.format(toxic),
                            pred_severe = 'Severe Toxic: {}'.format(severe_toxic), 
                            pred_obscene = 'Obscene: {}'.format(obscene),
                            pred_insult = 'Insult: {}'.format(insult),
                            pred_threat = 'Threat: {}'.format(threat),
                            pred_identity = 'Identity Hate: {}'.format(identity_hate)                        
                            )
     
# debut=True to reload the server itself
app.run(debug=True)
