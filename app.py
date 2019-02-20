#coding: utf - 8

# # Meet Robo: your friend

import nltk
nltk.download('punkt')
import warnings
from flask import Flask, render_template, request
nltk.data.path.append(r'heroku-python-script-master/nltk/')

warnings.filterwarnings("ignore")

# nltk.download() # for downloading packages

import numpy as np
import random
import string  # to process standard python strings

f = open('chatbot.txt', 'r', errors='ignore')
raw = f.read()
raw = raw.lower()  # converts to lowercase
# nltk.download('punkt') # first-time use only
# nltk.download('wordnet') # first-time use only
sent_tokens = nltk.sent_tokenize(raw)  # converts to list of sentences
word_tokens = nltk.word_tokenize(raw)  # converts to list of words


lemmer = nltk.stem.WordNetLemmatizer()


def LemTokens(tokens):
    lemma = [lemmer.lemmatize(token) for token in tokens]
    return lemma


remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)


def LemNormalize(text):
    LemmaNor = LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))
    #print(LemmaNor)
    return LemmaNor


GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]


# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Generating response
def response(user_response):
    robo_response = ''
    sent_tokens.append(user_response)
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    tfidf = TfidfVec.fit_transform(sent_tokens)
    vals = cosine_similarity(tfidf[-1], tfidf)
    idx = vals.argsort()[0][-2]
    flat = vals.flatten()
    flat.sort()
    req_tfidf = flat[-2]
    if (req_tfidf == 0):
        robo_response = robo_response + "I am sorry! I don't understand you"
        return robo_response
    else:
        robo_response = robo_response + sent_tokens[idx]
        return robo_response


class MyChatBOT:

    @staticmethod
    def get_response(user_response):
        if (user_response != 'bye'):
            if (user_response == 'thanks' or user_response == 'thank you'):
                return "ROBO: You are welcome.."
            else:
                if (greeting(user_response) != None):
                    return "ROBO: " + greeting(user_response)
                else:
                    #sent_tokens.remove(user_response)
                    return response(user_response)
        else:
            return "ROBO: Bye! take care.."

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

english_bot = MyChatBOT()

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return str(english_bot.get_response(userText))


if __name__ == "__main__":
    app.run()
