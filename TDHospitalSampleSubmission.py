# Sample participant submission for testing
from flask import Flask, jsonify, request, render_template
import pandas as pd
import random
import pickle
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)
with open('encoder.pkl', 'rb') as encoder_file:
    encoder = pickle.load(encoder_file)

class Solution:
    def __init__(self):
        #Initialize any global variables here
        with open('pickle_model.pkl', 'rb') as file:
            self.model = pickle.load(file)

    def calculate_death_prob(self, timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer):
        
        """
        This function should return your final prediction!
        """
        # print("called")
        labels = ["timeknown", "cost", "reflex", "sex", "blood", "bloodchem1", "bloodchem2", "temperature", "race",
 "heart", "psych1", "glucose", "psych2", "dose", "psych3", "bp", "bloodchem3", "confidence", "bloodchem4",
 "comorbidity", "totalcost", "breathing", "age", "sleep", "dnr", "bloodchem5", "pdeath", "meals", "pain",
 "primary", "psych4", "disability", "administratorcost", "urine", "diabetes", "income", "extraprimary",
 "bloodchem6", "education", "psych5", "psych6", "information", "cancer"]
        # check if a character is in each variable

        values = []
        for x in [timeknown, cost, reflex, sex, blood, bloodchem1, bloodchem2, temperature, race,
                             heart, psych1, glucose, psych2, dose, psych3, bp, bloodchem3, confidence, bloodchem4,
                             comorbidity, totalcost, breathing, age, sleep, dnr, bloodchem5, pdeath, meals, pain,
                             primary, psych4, disability, administratorcost, urine, diabetes, income, extraprimary,
                             bloodchem6, education, psych5, psych6, information, cancer]:
            try:
                values.append(float(x))
            except:
                values.append(x)

        df = dict()
        #print(len(labels))
        #print(len(values))
        for label, value in zip(labels, values):
            df[label] = [value]
        df = pd.DataFrame(df)
        df.replace('', 0, inplace=True)
        df.fillna(0, inplace=True)

        X = df
        # print(X)
        X = X.drop(X[X['race'] == 0].index)
        X = X.drop(X[X['dnr'] == 0].index)
        #X = X.drop(columns=['dose'])
        X['sex'] = X['sex'].replace(['M', 'Male'], 'male')
        X = X.drop('sex', axis=1)
        # X.head()
        df = X
        # print(X)
        # with open('scaler.pkl', 'rb') as scaler_file:
        #     scaler = pickle.load(scaler_file)
        # with open('encoder.pkl', 'rb') as encoder_file:
        #     encoder = pickle.load(encoder_file)
        X_numeric = scaler.transform(X.select_dtypes(include=['float64']))
        X[X.select_dtypes(include=['float64']).columns] = X_numeric
        # X = pd.get_dummies(X, columns = ['race', 'dnr', 'primary', 'disability', 'income', 'extraprimary', 'cancer'])
        # print(X)
        X = encoder.transform(X)
        #print(X)
        val = self.model.predict(X[0].reshape(1, -1))[0]
        
        
        return float(val)

@app.route("/")
def hello():
    return render_template('index.html')

# BOILERPLATE
@app.route("/death_probability", methods=["POST"])
def q1():
    # print("server received request")
    solution = Solution()
    data = request.get_json()
    # print("server received data", data)
    allEntries = ["timeknown", "cost", "reflex", "sex", "blood", "bloodchem1", "bloodchem2", "temperature", "race",
                             "heart", "psych1", "glucose", "psych2", "dose", "psych3", "bp", "bloodchem3", "confidence", "bloodchem4",
                             "comorbidity", "totalcost", "breathing", "age", "sleep", "dnr", "bloodchem5", "pdeath", "meals", "pain",
                             "primary", "psych4", "disability", "administratorcost", "urine", "diabetes", "income", "extraprimary",
                             "bloodchem6", "education", "psych5", "psych6", "information", "cancer"]
    # for entry in allEntries:
    #     if entry not in data:
    #         data[entry] = 0
    return {
        "probability": solution.calculate_death_prob(data['timeknown'], data['cost'], data['reflex'], data['sex'], data['blood'],
                                            data['bloodchem1'], data['bloodchem2'], data['temperature'], data['race'],
                                            data['heart'], data['psych1'], data['glucose'], data['psych2'],
                                            data['dose'], data['psych3'], data['bp'], data['bloodchem3'],
                                            data['confidence'], data['bloodchem4'], data['comorbidity'],
                                            data['totalcost'], data['breathing'], data['age'], data['sleep'],
                                            data['dnr'], data['bloodchem5'], data['pdeath'], data['meals'],
                                            data['pain'], data['primary'], data['psych4'], data['disability'],
                                            data['administratorcost'], data['urine'], data['diabetes'], data['income'],
                                            data['extraprimary'], data['bloodchem6'], data['education'], data['psych5'],
                                            data['psych6'], data['information'], data['cancer'])}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5555)
