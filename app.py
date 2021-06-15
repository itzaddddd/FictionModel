from flask import Flask, request, jsonify, make_response
from flask_restful import Api
from pythainlp import word_tokenize
from pythainlp.util import normalize, dict_trie
from pythainlp.corpus import thai_stopwords, thai_words, thai_female_names, thai_male_names, wordnet
from nltk.stem.porter import PorterStemmer
import joblib
import dill as pickle
import re
import string
import emoji
import pandas as pd
nltk.download()

app = Flask(__name__)
api = Api(app)

# Load Model
clf_path = 'models/fic-model.pkl'
with open(clf_path,'rb') as f:
    model = joblib.load(f)

# Load Tfidf
tfidf_path = 'models/fic-tfidf.pkl'
with open(tfidf_path,'rb') as f:
    tfidf = pickle.load(f)

@app.route("/")
def home():
    return make_response("Fiction Gene Model Service", 200)

@app.route("/predict", methods=['POST','GET'])
def prediction():
    if request.method == 'GET':
        return make_response('Resend with POST Method', 200)
    
    elif request.method == 'POST':
        x = request.get_json()
        x_content = x['content']

        x_text = re.sub(r'<.*?>','', x_content)
        for c in string.punctuation:
            x_text = re.sub(r'\{}'.format(c),'',x_text)

        x_text = re.sub(r'[-()!\'"#/@;:<>{}=~|.?,“”‘’\n\tๆA-Za-z0-9…]','', x_text)
        x_text = ' '.join(x_text.split())    
        x_text = ''.join([c for c in x_text if c not in emoji.UNICODE_EMOJI])    
        x_text = normalize(x_text)

        my_character_name = pd.read_csv('./models/character_name_list.csv', encoding = 'utf-8', sep=',')
        my_character_name = my_character_name.stack().tolist()
        name_list_fe = set(thai_female_names())
        name_list = set(thai_male_names())
        name_list.update(name_list_fe)
        name_list.update(my_character_name)

        my_stopwords = pd.read_csv('./models/my_stopwords.csv', encoding = 'utf-8', sep=',')
        my_stopwords = my_stopwords.stack().tolist()
    
        df_spell_check = pd.read_csv('./models/word_adding.csv', encoding = 'utf-8', sep=',')
        list_df_spell_check = df_spell_check.stack().tolist()
        
        custom_stopwords_list = set(thai_stopwords())
        custom_stopwords_list.update(my_stopwords)
        check_unknown_stopword2 = [i for i in list_df_spell_check if i not in custom_stopwords_list]

        custom_words_list = set(thai_words())
        check_unknown_word2 = [i for i in check_unknown_stopword2 if i not in custom_words_list]
        custom_words_list.update(check_unknown_word2)

        trie = dict_trie(dict_source=custom_words_list)
    
        tokens = word_tokenize(x_text, custom_dict=trie, engine='newmm', keep_whitespace=False)
        tokens = [i for i in tokens if i not in custom_stopwords_list]
        tokens = [i for i in tokens if i not in name_list]

        p_stemmer = PorterStemmer()
        tokens = [p_stemmer.stem(i) for i in tokens]
        
        tokens_temp=[]
        for i in tokens:
            w_syn = wordnet.synsets(i)
            if (len(w_syn)>0) and (len(w_syn[0].lemma_names('tha'))>0):
                tokens_temp.append(w_syn[0].lemma_names('tha')[0])
            else:
                tokens_temp.append(i)

        tokens = tokens_temp    
        tokens = [i for i in tokens if not i.isnumeric()]   
        tokens = [i for i in tokens if not ' ' in i]

        x_trans = tfidf.transform([tokens])
        
        prob_raw = model.predict_proba(x_trans).round(4)
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
    app.run()
