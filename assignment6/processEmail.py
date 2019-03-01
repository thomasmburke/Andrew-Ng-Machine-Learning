import re
from nltk.stem import PorterStemmer

def processEmail(email):
    email = email.lower()
    # Replace all numbers with 'number'
    email = re.sub(r'[0-9]+', 'number', email)
    email = re.sub(r'(http|https)://[^\s]*','httpaddr', email)
    words = get_vocab_list()
    return email

def get_vocab_list():
    with open('data/vocab.txt', mode='r') as myFile:
        words = myFile.read().splitlines()
    pattern = r'(\d+)\t(\w+)'
    cleansedWords = {}
    for word in words:
        match = re.search(pattern, word)
        cleansedWords.update({match.group(2): match.group(1)})
    return cleansedWords
