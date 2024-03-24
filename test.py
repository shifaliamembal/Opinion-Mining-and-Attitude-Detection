import pandas as pd
from flask import Flask, render_template, request,redirect,url_for

app = Flask(__name__)

time = ['Tue Mar 05 06:12:41 +0000 2024','Tue Mar 05 07:09:46 +0000 2024','Tue Mar 05 07:49:55 +0000 2024']
only_tweets = ["Congress party has put such posters of Modi Ka Asli Parivar all across Delhi overnight. Brilliant move by Congress party to expose BJP's propaganda, this is a huge sixer ", "As soon as our government comes, we will guarantee MSP to the farmers. #BharatJodoNyayYatra", "Supreme court dismisses money laundering case against Karnataka Deputy Chief minister of Karnataka Dk Shivakumar. This now BJP misuse agency against Congress leader!"]
sent = ["Pos", "Neg", "Neutral"]

tweet = pd.DataFrame(only_tweets, columns=['Tweet'])
time = pd.DataFrame(time, columns=['Time'])
sentiment_col = pd.DataFrame(sent, columns=['Sentiment'])
df = pd.concat([time,tweet, sentiment_col], axis=1)
# styled_df = df.style.set_properties(**{'max-width': '100%'})
styled_html = df.style.set_table_styles([
    {
    'selector': '.row_heading, .blank',
    'props': [('display', 'none')]
    },
    {
        'selector': 'table',
        'props': [('border-collapse', 'collapse')]
    },
    {
        'selector': 'th, td',
        'props': [('border', '1px solid black'), ('padding', '8px')]
    },
    {
        'selector': 'th',
        'props': [('background-color', '#f2f2f2')]
    }
]).render()


@app.route('/', methods=['GET', 'POST'])
def table1():
    return render_template('dataframe.html',data=styled_html)


if __name__ == '__main__':
    app.run(debug=True)