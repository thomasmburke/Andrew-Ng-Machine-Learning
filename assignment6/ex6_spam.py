import numpy as np
from scipy.io import loadmat
from sklearn.svm import SVC
"""
%% Machine Learning Online Class
%  Exercise 6 | Spam Classification with SVMs
%
%  Instructions
%  ------------
% 
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     gaussianKernel.m
%     dataset3Params.m
%     processEmail.m
%     emailFeatures.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%% ==================== Part 1: Email Preprocessing ====================
%  To use an SVM to classify emails into Spam v.s. Non-Spam, you first need
%  to convert each email into a vector of features. In this part, you will
%  implement the preprocessing steps for each email. You should
%  complete the code in processEmail.m to produce a word indices vector
%  for a given email.
"""
print('Preprocessing sample email (emailSample1.txt)')
with open('data/emailSample1.txt', mode='r') as myFile:
    emailSample1 = myFile.read()
from processEmail import processEmail, get_vocab_list
processedEmail1 = processEmail(emailSample1)
print(processedEmail1)
uniqueWords = get_vocab_list()
#print('number of unique words: {}'.format(uniqueWords))
"""
%% ==================== Part 2: Feature Extraction ====================
%  Now, you will convert each email into a vector of features in R^n. 
%  You should complete the code in emailFeatures.m to produce a feature
%  vector for a given email.
"""
print('Extracting features from sample email (emailSample1.txt)')
from emailFeatures import emailFeatures
# Extract Features
features = emailFeatures(processedEmail1)

# Print Stats
print('Length of feature vector: {}'.format(len(features)))
print('Number of non-zero entries: {}'.format(sum(features > 0)))
"""
%% =========== Part 3: Train Linear SVM for Spam Classification ========
%  In this section, you will train a linear classifier to determine if an
%  email is Spam or Not-Spam.
"""
# Load the Spam Email dataset
# You will have X, y in your environment
data = loadmat('data/spamTrain.mat')
X = data['X']
y = data['y']
m, n = X.shape
print('Training Linear SVM (Spam Classification)')
print('(this may take 1 to 2 minutes) ...')
classifier = SVC(C=0.1, kernel='linear')
classifier.fit(X,np.ravel(y))
print("Training Accuracy:",(classifier.score(X,y.ravel()))*100,"%")
"""
%% =================== Part 4: Test Spam Classification ================
%  After training the classifier, we can evaluate it on a test set. We have
%  included a test set in spamTest.mat
"""
#Load the test dataset
# You will have Xtest, ytest in your environment
testData = loadmat('data/spamTest.mat')
X_test = testData['Xtest']
y_test = testData['ytest']

print('Evaluating the trained Linear SVM on a test set ...')
print("Training Accuracy:",(classifier.score(X_test,y_test.ravel()))*100,"%")

"""
%% ================= Part 5: Top Predictors of Spam ====================
%  Since the model we are training is a linear SVM, we can inspect the
%  weights learned by the model to understand better how it is determining
%  whether an email is spam or not. The following code finds the words with
%  the highest weights in the classifier. Informally, the classifier
%  'thinks' that these words are the most likely indicators of spam.
"""
weights = classifier.coef_[0]
weights = np.hstack((np.arange(1,1900).reshape(1899,1),weights.reshape(1899,1)))
print(weights)
print(weights.shape)
indexes = weights[weights[:,1].argsort()][-15:,0]
print(indexes)
intIndexes = []
for index in indexes:
    intIndexes.append(int(index))
print(intIndexes)
topSpamWords = []
for word, index in uniqueWords.items():
    for intIndex in intIndexes:
        if str(intIndex) == str(index):
            topSpamWords.append(word)
print(topSpamWords)
"""
# Sort the weights and obtin the vocabulary list
[weight, idx] = sort(model.w, 'descend');
vocabList = getVocabList();

fprintf('\nTop predictors of spam: \n');
for i = 1:15
    fprintf(' %-15s (%f) \n', vocabList{idx(i)}, weight(i));
end

fprintf('\n\n');
fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% =================== Part 6: Try Your Own Emails =====================
%  Now that you've trained the spam classifier, you can use it on your own
%  emails! In the starter code, we have included spamSample1.txt,
%  spamSample2.txt, emailSample1.txt and emailSample2.txt as examples. 
%  The following code reads in one of these emails and then uses your 
%  learned SVM classifier to determine whether the email is Spam or 
%  Not Spam

% Set the file to be read in (change this to spamSample2.txt,
% emailSample1.txt or emailSample2.txt to see different predictions on
% different emails types). Try your own emails as well!
filename = 'spamSample1.txt';

% Read and predict
file_contents = readFile(filename);
word_indices  = processEmail(file_contents);
x             = emailFeatures(word_indices);
p = svmPredict(model, x);

fprintf('\nProcessed %s\n\nSpam Classification: %d\n', filename, p);
fprintf('(1 indicates spam, 0 indicates not spam)\n\n');
"""
