import re

def processEmail(email):
    email = email.lower()
    words = get_vocab_list()
    print(words)
    return email

def get_vocab_list():
    with open('data/vocab.txt', mode='r') as myFile:
        words = myFile.read().splitlines()
    pattern = r'(\d+\t)(\w+)'
    cleansedWords = []
    for word in words:
        cleansedWords.append(re.search(pattern, word).group(2))
    return cleansedWords
