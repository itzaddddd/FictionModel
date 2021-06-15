import re
import ast
import emoji
import string
import joblib
import numpy as np 
import pandas as pd
import dill as pickle
from pythainlp import word_tokenize
from pythainlp.util import normalize, dict_trie
from pythainlp.corpus import thai_stopwords, thai_words, thai_female_names, thai_male_names, wordnet
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

# Read File
file = pd.read_csv('df_all_training_novel.csv',encoding='utf-8')
file.drop(columns='Unnamed: 0', inplace=True)
file.head()

# Retrieve List from String
file['Chapter'] = file['Chapter'].apply(lambda x: ast.literal_eval(x))

def dummy_fun(doc):
    return doc

# Create Tfidf Vect
tfidf = TfidfVectorizer(
    tokenizer=dummy_fun,
    preprocessor=dummy_fun,
    token_pattern=None,
    max_features=5000,
    min_df=0.01
)

# Transform to Tfidf Vect
X = tfidf.fit_transform(file['Chapter'])
# Set y
y = file['Genre']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=0)

# Create Naive Bayes Model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Evaluate Performance
y_pred = nb.predict(X_test)
f1_sc = np.round(f1_score(y_test, y_pred, average='macro'),2)
ac_sc = np.round(accuracy_score(y_test, y_pred),2)
print('F1 score = ',f1_sc)
print('Accuracy = ',ac_sc)

################################################################################

# Getiing features name
# response = tfidf.transform([ex])
# feature_array = np.array(tfidf.get_feature_names())
# tfidf_sorting = np.argsort(response.toarray()).flatten()[::-1]

# n = 15
# top_n = feature_array[tfidf_sorting][:n]


################################################################################
def showProb(text):
   
    text = re.sub(r'<.*?>','', text)      
    for c in string.punctuation:
        text = re.sub(r'\{}'.format(c),'',text)
    
    text = re.sub(r'[-()!\'"#/@;:<>{}=~|.?,“”‘’\n\tๆA-Za-z0-9…]','', text)
    text = ' '.join(text.split())    
    text = ''.join([c for c in text if c not in emoji.UNICODE_EMOJI])    
    text = normalize(text)
    
    my_character_name = pd.read_csv('character_name_list.csv', encoding = 'utf-8', sep=',')
    my_character_name = my_character_name.stack().tolist()
    name_list_fe = set(thai_female_names())
    name_list = set(thai_male_names())
    name_list.update(name_list_fe)
    name_list.update(my_character_name)
    
    my_stopwords = pd.read_csv('my_stopwords.csv', encoding = 'utf-8', sep=',')
    my_stopwords = my_stopwords.stack().tolist()
    
    df_spell_check = pd.read_csv('word_adding.csv', encoding = 'utf-8', sep=',')
    list_df_spell_check = df_spell_check.stack().tolist()
    
    custom_stopwords_list = set(thai_stopwords())
    custom_stopwords_list.update(my_stopwords)
    check_unknown_stopword2 = [i for i in list_df_spell_check if i not in custom_stopwords_list]

    custom_words_list = set(thai_words())
    check_unknown_word2 = [i for i in check_unknown_stopword2 if i not in custom_words_list]
    custom_words_list.update(check_unknown_word2)
    
    trie = dict_trie(dict_source=custom_words_list)
    
    tokens = word_tokenize(text, custom_dict=trie, engine='newmm', keep_whitespace=False)
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
    
    # Test model
    ex_tk_trans = tfidf.transform([tokens])
    result = nb.predict(ex_tk_trans)
    prob = nb.predict_proba(ex_tk_trans).round(2)
    pdf = pd.DataFrame(data=prob,columns=nb.classes_)
    
    print('Result : ',result[0])
    
    return pdf

################################################################################

# Test Predict Example with text file
with open('testing novel/ดราม่า/รักครั้งสุดท้ายผ่านรูป(ใบเก่า)/ตอนที่ 1.txt', 'r', encoding="utf-8") as f:
    Xtest = f.read()

ex = Xtest

print(showProb(ex))

################################################################################

# Save Tfidf Vectorizer
tfidf_path = "fic-tfidf.pkl"
pickle.dump(tfidf, open(tfidf_path,"wb"))
# Save Model
model_path = "fic-model.pkl"
joblib.dump(nb, model_path)



