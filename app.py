from flask import Flask, render_template, request,redirect,url_for
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

import re
import pickle
import pandas as pd

from apify_client import ApifyClient

# Initialize the ApifyClient with your API token
client = ApifyClient("apify_api_ipuc2MyZvMcB9xNND443iQx6ARFh861h7zUb")

def preprocess_text(df, column_name):
    # Make a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()

    # Apply the text preprocessing steps
    df_copy[column_name] = df_copy[column_name] \
        .str.replace(r'(?:@|#|https?:|www\.)\S+', '') \
        .str.replace(r'[^A-Za-z0-9 ]+', '') \
        .str.split() \
        .str.join(' ') \
        .str.lower()
    return df_copy

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
# Load model
model = load_model('best_model.h5')

def predict_class(text):
    '''Function to predict sentiment class of the passed text'''
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len=50
    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    sentiment = sentiment_classes[yt[0]]
    return sentiment

test_data = []

app = Flask(__name__)

@app.route('/', methods=['GET','POST'])
def index():
    return render_template('home.html')

@app.route('/custom', methods=['GET','POST'])
def predictmytweet():
    if request.method == 'POST':
        text = request.form['tweet']
        test_data.insert(0,text)
        df = pd.DataFrame({'tweets': test_data})
        preprocessed_df = preprocess_text(df, 'tweets')
        prediction = predict_class(preprocessed_df['tweets'])
        return render_template('index.html', t=prediction)
    else:
        return render_template('index.html')

@app.route('/topics', methods=['GET', 'POST'])
def topics():
    return render_template('index2.html')


@app.route('/success', methods=['GET', 'POST'])
def success():
    topic = request.form['topic']
    return redirect(url_for('dataframe', topic=topic))


@app.route('/dataframe/<topic>', methods=['GET','POST'])
def dataframe(topic):
    # Use snscrape to search Twitter for recent tweets related to the topic
    time = []
    only_tweets = []
    predict = []
    cleaned = []
    # Prepare the Actor input
    run_input = {
        "searchTerms": [topic],
        "searchMode": "top",
        "maxTweets": 5,
        "addUserInfo": False
    }
    pattern = r'https?://\S+'
    # Run the Actor and wait for it to finish
    run = client.actor("heLL6fUofdPgRXZie").call(run_input=run_input)
    # Fetch and print Actor results from the run's dataset (if there are any)
    for item in client.dataset(run["defaultDatasetId"]).iterate_items():
        #tweets.append([item["created_at"],item["full_text"]])
        time.append(item['created_at'])
        only_tweets.append(item['full_text'])
    
    for text in only_tweets:
        text1 = re.sub(pattern, '', text)
        cleaned.append(text1)
        test_data.insert(0,text1)
        df = pd.DataFrame({'tweets': test_data})
        preprocessed_df = preprocess_text(df, 'tweets')
        prediction = predict_class(preprocessed_df['tweets'])
        predict.append(prediction)
    
    # Convert the results into a pandas DataFrame
    tweet = pd.DataFrame(cleaned, columns=['Tweet'])
    time = pd.DataFrame(time, columns=['Time'])
    sentiment_col = pd.DataFrame(predict, columns=['Sentiment'])
    df = pd.concat([time,tweet, sentiment_col], axis=1)
    styled_df = df.style.set_properties(**{'max-width': '100%'})
    styled_html = styled_df.to_html()
    # styled_html = df.style.set_table_styles([
    # {
    # 'selector': '.row_heading, .blank',
    # 'props': [('display', 'none')]
    # },
    # {
    #     'selector': 'table',
    #     'props': [('border-collapse', 'collapse')]
    # },
    # {
    #     'selector': 'th, td',
    #     'props': [('border', '1px solid black'), ('padding', '8px')]
    # },
    # {
    #     'selector': 'th',
    #     'props': [('background-color', '#f2f2f2')]
    # }
    # ]).render()
    # Render the DataFrame as an HTML table
    return render_template('dataframe.html',topic=topic,data=styled_html)


if __name__ == '__main__':
    app.run(debug=True)

