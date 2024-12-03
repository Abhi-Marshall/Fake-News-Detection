from flask import Flask, render_template, request
import joblib
from feature import *

# Load the saved pipeline
pipeline = joblib.load('./pipeline.sav')

app = Flask(__name__)

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction API
@app.route('/api', methods=['POST'])
def get_prediction():
    result = request.form
    query_title = result['title']
    query_author = result['author']
    query_text = result['maintext']

    # Prepare the query
    query = get_all_query(query_title, query_author, query_text)
    
    # Process the query
    query = remove_punctuation_stopwords_lemma(query[0])
    
    # Predict the result
    pred = pipeline.predict([query])
    dic = {1: 'real', 0: 'fake'}
    
    # Render the result in the template
    return render_template('result.html', prediction=dic[pred[0]])

# Main block to run the server
if __name__ == '__main__':
    app.run(port=8080, debug=True)
