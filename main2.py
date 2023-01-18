from flask import Flask, jsonify, request, json
import numpy as np
import pickle
import re
from keras_preprocessing.sequence import pad_sequences
import tensorflow as tf


# sales_model = Blueprint('sales_model', __name__)

sales_model = Flask(__name__)


sales_vs_generic = tf.keras.models.load_model(r"C:\Users\Rafay\Downloads\sales_vs_nonsales2\sales_vs_nonsales2\99_accuracy_lstm.h5")

with open(r'C:\Users\Rafay\Downloads\sales_vs_nonsales2\sales_vs_nonsales2\tokenizer.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

def clean(text):
    text = text.lower()
    text = re.sub("(http|https|www)(:|\.)\S+.com"," ",text)
    text = re.sub("[^\w\d]"," ",text)
    text = " ".join([t for t in text.split()])
    return text

def predict_sales_non_sales(model,input_text, tokenizer):

    string = clean(input_text)
    output = tokenizer.texts_to_sequences([string])
    output = pad_sequences(output,padding='pre',truncating='pre',maxlen=30)

    temp = np.reshape(output, (-1, output.size))
    output = model.predict(temp)
    return output


@sales_model.route('/', methods=['POST'])
def sales_classifier():
    try:
        if request.method == 'POST':		
            event = json.loads(request.data)
            new_sentence = event['input']


            try:
                output = predict_sales_non_sales(sales_vs_generic, new_sentence, tokenizer1)
            except Exception as e:
                return jsonify({"ERROR in prediction": str(e)})
            prediction = 'sales' if round(output[0][0]) == 1 else 'non_sales'

            print('Predicted Class: ', prediction)

            data = {}
            data['output'] = prediction
            json_data = json.dumps(data)

            return json_data
    except Exception as e:
        msg = str(e)
        return jsonify({"ERROR": msg})

if __name__ == '__main__':
    # starting app
   sales_model.run(debug=True)
