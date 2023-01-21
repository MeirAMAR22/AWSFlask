import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)


def read_pkl(filename):
    """Import the model to predict the target"""
    with open(f'C:/Users/Meir/OneDrive/ITC2/COURSES/FLASK/Ex2/{filename}.pkl', 'rb') as file:
        model = pickle.load(file)
        return model


@app.route('/predict_single', methods=['GET'])
def predict_single():
    """Perform the prediction using an API"""
    is_male = request.args.get('is_male')
    num_inters = request.args.get('num_inters')
    late_on_payment = request.args.get('late_on_payment')
    age = request.args.get('age')
    years_in_contract = request.args.get('years_in_contract')
    input_data = np.array([is_male, num_inters, late_on_payment, age, years_in_contract]).reshape(1, -1)
    print(input_data)
    # Perform the prediction using the inputs
    model = read_pkl('churn_model')
    my_pred = str(model.predict(input_data))
    return my_pred[1]


def main():
    """Group all the tasks in the right order"""
    app.run(host='0.0.0.0', port=8080, debug=True)


if __name__ == '__main__':
    main()
