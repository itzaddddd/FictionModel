from flask import Flask, request, jsonify, make_response
from flask_restful import Api
from pythainlp import word_tokenize
import joblib
import dill as pickle

app = Flask(__name__)
api = Api(app)

# Load Model
clf_path = 'lib/models/fic-model.pkl'
with open(clf_path,'rb') as f:
    model = joblib.load(f)

# Load Tfidf
tfidf_path = 'lib/models/fic-tfidf.pkl'
with open(tfidf_path,'rb') as f:
    tfidf = pickle.load(f)

@app.route("/")
def home():
    return "Fiction Gene Model Service"

@app.route("/predict", methods=['POST','GET'])
def prediction():
    if request.method == 'GET':
        return make_response('Resend with POST Method',200)
    
    elif request.method == 'POST':
        x = request.get_json()
        x_content = x['content']
        x_token = word_tokenize(x_content)
        x_trans = tfidf.transform([x_token])
        
        prob_raw = model.predict_proba(x_trans).round(2)
        prob = prob_raw[0]
        prob_genre = []
        for i in range(0, len(model.classes_)):
            sub = {
                'genre': model.classes_[i],
                'prob': prob[i]
            }
            prob_genre.append(sub)
        prob_genre.sort(key=lambda x:x['prob'],reverse=True)

        return make_response(jsonify(prob_genre), 200)

if __name__ == "__main__":
    app.run(debug=True)
