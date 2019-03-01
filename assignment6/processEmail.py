import re
from nltk.stem import PorterStemmer

def processEmail(email):
    # lowercase everything in the email
    email = email.lower()
    # Replace all numbers with 'number'
    email = re.sub(r'[0-9]+', 'number', email)
    # Clean up URLs
    email = re.sub(r'(http|https)://[^\s]*','httpaddr', email)
    # Clean up email addresses
    email = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email)
    # Clean up dollar signs $
    email = re.sub(r'[$]+','dollar', email)
    # Strip all special characters
    specialChar = ["<","[","^",">","+","?","!","'",".",",",":"]
    for char in specialChar:
        email = email.replace(str(char),'')
    # Remove newline characters and replace with space
    email = email.replace("\n"," ")
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
