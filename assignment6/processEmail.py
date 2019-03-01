import re
from nltk.stem import PorterStemmer

def processEmail(email):
    # Get dictionary of acceptable words
    words = get_vocab_list()
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
    email = email.replace('\n', ' ')
    # Stem the word
    ps = PorterStemmer()
    email = [ps.stem(token) for token in email.split()]
    email = ' '.join(email)
    # Process the email and return word_indices
    word_indices=[]
    for char in email.split():
        if len(char) > 1 and char in words:
            word_indices.append(int(words[char]))
    return word_indices

def get_vocab_list():
    with open('data/vocab.txt', mode='r') as myFile:
        words = myFile.read().splitlines()
    pattern = r'(\d+)\t(\w+)'
    cleansedWords = {}
    for word in words:
        match = re.search(pattern, word)
        cleansedWords.update({match.group(2): match.group(1)})
    return cleansedWords
