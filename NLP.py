from newsapi import NewsApiClient 
import spacy
from spacy.tokenizer import Tokenizer
import pandas as pd
from collections import Counter
from string import punctuation
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import pickle

nlp_eng = spacy.load('en_core_web_lg')
tokenizer = Tokenizer(nlp_eng.vocab)
newsapi = NewsApiClient(api_key='58e9d0461d6049e7a9fb51abf54055e0')
count = 1
articles = []

def get_keywords_eng(text):
    result = []
    pos_tag = ['PROPN', 'VERB', 'NOUN']
    doc = nlp_eng(text.lower())
    for token in doc:
        if (token.text in nlp_eng.Defaults.stop_words or token.text in punctuation):
            continue
        if (token.pos_ in pos_tag):
            result.append(token.text)
    return result

filename = 'articlesCOVID.pckl'
pickle.dump(articles, open(filename, 'wb'))

while count < 6:
    articles.append(newsapi.get_everything(q='coronavirus', language='en', from_param='2022-03-03', to='2022-03-24', sort_by='relevancy', page=count))
    count += 1

dados = {}
titles = []
dates = []
descriptions = []
content = []

for i, article in enumerate(articles):
    for x in article['articles']:
        titles.append(x['title'])
        descriptions.append(x['description'])
        dates.append(x['publishedAt'])
        content.append(x['content'])
        print(x['content'])
        print(len(tokenizer(x['content'])))
        dados.update({'title': titles, 'date': dates, 'desc': descriptions, 'content': content})
df = pd.DataFrame(dados)
df = df.dropna()
df.head()

results = []

for content in df.content.values:
    results.append([('#' + x[0]) for x in Counter(get_keywords_eng(content)).most_common(5)])

text = str(results)

text = ""
for x in results:
    for y in x:
        text += " " + y

df['keywords'] = results

df.to_csv('dataset.csv')
print(results)

wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()