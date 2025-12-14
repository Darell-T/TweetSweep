from datasets import load_dataset
from datasets import concatenate_datasets
import re
import ftfy
import html
from profanity_check import predict, predict_prob


#dataset #1 hate speech
ds = load_dataset("tweets-hate-speech-detection/tweets_hate_speech_detection")
ds2 = load_dataset("Masabanees619/toxic_tweets_and_comments")
ds3 = load_dataset("Om1024/racist-sexist")



#print(ds["train"].column_names)
#label 0 == normal, 1== hate speech

def clean_tweets(text):
    
    text = ftfy.fix_text(text) #fix the stupid broken emojis
    text = html.unescape(text)
    text = re.sub(r'http\S+|www\.\S+', '[URL]', text)

    text = re.sub(r'\s+', ' ', text)
    return text

def map_hate_speech_tweets(text):

    tweet_text = text['tweet']
    is_hate_speech = text['label']

    return {
        'text': clean_tweets(tweet_text),
        'hate_speech': is_hate_speech,
        'profanity': predict([tweet_text]),
        'toxic': is_hate_speech
    }

def map_racist_sexist_tweets(text):
    tweet_text = text['text']
    is_racist = text['racist'] == 1
    is_sexist = text['sexist'] == 1

    is_hate = is_racist or is_sexist
    return {
        'text': clean_tweets(tweet_text),
        'hate_speech': 1 if is_hate else 0,
        'toxic': 1 if is_hate else 0,
        'profanity': predict([tweet_text])
    }

def map_toxic_tweets(text):
    tweet_text = text['clean_text']
    is_toxic = 1
    return{
        'text': clean_tweets(tweet_text),
        'hate_speech': 0,
        'toxic': is_toxic,
        'profanity': predict([tweet_text])
    }
