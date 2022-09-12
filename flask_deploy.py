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


    text = userText.lower()
    text = re.sub("[0-9]","", text)
    text = text.replace('?', '')
    text = text.replace(',', '')
    text = re.sub("  "," ", text)
    
    userTextClean = [] 
    for word in text.split(' '):
        if word not in stopwords:
            userTextClean.append(word)
    
    lemmaText = []
    for word in userTextClean:
        lemmaText.append(lemmatizer.lemmatize(word.strip('"')))

    finalText = ' '.join(lemmaText)

    sequenced_input = tokenizer.texts_to_sequences([finalText])
    padded_input = pad_sequences(sequenced_input, truncating='post', maxlen=10)
    result = model.predict(padded_input)

    results = {}
    for i in range(len(result[0])):
        results[i] = model.predict(padded_input)[0][i]
    sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
    list_results = list(sorted_results)
    predicted_class_index = int(list_results[0])
    return index_responses.get(predicted_class_index)


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True, threaded=True)