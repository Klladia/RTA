import pickle
from math import log10
from flask import Flask
from flask import request
from flask import jsonify
from sklearn import linear_model
from sklearn.externals import joblib




app = Flask(__name__)
Model = joblib.load('Perceptron_model.pkl')




@app.route('/api/v1.0/predict', methods=['GET'])
def get_prediction():

    s_length = float(request.args.get('sepal_l'))
    p_length = float(request.args.get('petal_l'))

    s_width = float(request.args.get('sepal_w'))
    p_width = float(request.args.get('petal_w'))

    features = [s_length,
                s_width,
                p_length,
                p_width]


    predicted_class = int(Model.predict([features]))

    return jsonify(features=features, predicted_class=predicted_class)


if __name__ == '__main__':
    app.run(port=3333,host='0.0.0.0')