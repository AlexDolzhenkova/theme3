import flask
from flask import render_template
from tensorflow.keras.models import load_model
import os
import pickle
import sklearn
from sklearn.neural_network import MLPRegressor
from werkzeug.debug import DebuggedApplication

app = flask.Flask(__name__, template_folder = 'templates')

@app.route('/', methods = ['POST', 'GET'])
@app.route('/index', methods = ['POST', 'GET'])

def main():
    print('Мой путь:', os.path.dirname(__file__))
    if flask.request.method == 'GET':
        return render_template('main.html')

    if flask.request.method == 'POST':
        with open('mlpr_model.pkl', 'rb') as w:
            loaded_model_Width = pickle.load(w)
        with open('mlpr_model2.pkl', 'rb') as d:
            loaded_model_Depth = pickle.load(d)
        #loaded_model_Depth = pickle.load(os.path.join(os.path.dirname(__file__),'models', 'mlpr_model2.pkl', 'rb'))
        #loaded_model_Width = pickle.load(os.path.join(os.path.dirname(__file__),'models', 'mlpr_model.pkl', 'rb'))
        IW = float(flask.request.form['IW'])
        IF = float(flask.request.form['IF'])
        VW = float(flask.request.form['VW'])
        FP = float(flask.request.form['FP'])
        #Depth = float(flask.request.form['Depth'])
        #Width = float(flask.request.form['Width'])

        y_pred_Depth = loaded_model_Depth.predict([[IW, IF, VW, FP]])
        y_pred_Width = loaded_model_Width.predict([[IW, IF, VW, FP]])

        return render_template('main.html', result_depth = y_pred_Depth, result_width = y_pred_Width)

if __name__ == '__main__':
    app.debug = True
    app.run(host='0.0.0.0')