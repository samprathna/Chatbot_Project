from flask import Flask, render_template, request
import pickle
import nltk 
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import re

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/get')
def get_bot_response():
    userText = request.args.get('msg')
    
    tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))
    model = load_model('chatbotmodel.h5')
    index_responses = pickle.load(open('index_responses.pkl', 'rb'))
    stopwords = pickle.load(open('stopwords.pkl', 'rb'))
    lemmatizer = pickle.load(open('lemmatizer.pkl', 'rb'))

    def lowerText(InputString):
        return_1 = InputString.lower()
        return return_1
        
    def subNumbers(return_1):
        return_2 = re.sub("[0-9]","", return_1)
        return return_2

    def subQues(return_2):
        return_3 = return_2.replace('?', '')
        return return_3
        
    def subDoubleSpaces(return_3):
        return_4 = re.sub("  "," ", return_3)
        return return_4
        
    def subCommas(return_4):
        return_5 = return_4.replace(',', '')
        return return_5
        
    def removeStopwords(return_5):
        return_6 = [] 
        for word in return_5.split(' '):
            if word not in stopwords:
                return_6.append(word)
        return return_6

    def lemma(return_6):
        return_7 = []
        for word in return_6:
            return_7.append(lemmatizer.lemmatize(word.strip('"')))
        return return_7

    def toString(return_7):
        return_8 = ' '.join(return_7)
        return return_8
        
    def toSeq(return_8):
        return_9 = tokenizer.texts_to_sequences([return_8])
        return return_9
        
    
    def padSeq(return_9):
        return_10 = pad_sequences(return_9, truncating='post', maxlen=10)
        return return_10
        
    userText = lowerText(userText)
    userText = subNumbers(userText)
    userText = subQues(userText)
    userText = subDoubleSpaces(userText)
    userText = subCommas(userText)
    userText = removeStopwords(userText)
    userText = lemma(userText)
    userText = toString(userText)
    userText = toSeq(userText)
    userText = padSeq(userText)

    
    result = model.predict(userText)

    results = {}
    for i in range(len(result[0])):
        results[i] = result[0][i]
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    list_results = list(sorted_results)
    predicted_class_index = int(list_results[0])
    
    return index_responses.get(predicted_class_index) # Here do I need to transform my index_responses file to JSON?


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)