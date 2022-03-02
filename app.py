 
from flask import Flask, request
from src.utils import config
from src.ML.models import Models
import json
import tensorflow as tf
from tensorflow  import keras
import numpy as np

import torch
from src.word2vec.autoencoder_pytorch import Vocab, Seq2SeqEncoder

 
global graph, sess

graph = tf.compat.v1.get_default_graph()
sess = tf.compat.v1.keras.backend.get_session()
       
 
model = Models(model_path=config.root_path + '/model/ml_model/lightgbm.pkl', train_mode=False)



app = Flask(__name__)
 


@app.route('/predict', methods=["POST"])
def classification():
     
    result = {}
    
    #desc + title    
    text_ = request.form['text_']    
    
    with sess.as_default():
        with graph.as_default():
            label, score = model.predict(text_)
    result = {
        "label": label,
        "proba": str(score)
    }
    return json.dumps(result, ensure_ascii=False)


 
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)