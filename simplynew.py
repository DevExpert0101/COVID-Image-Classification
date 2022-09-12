# -*- coding: utf-8 -*-
import tensorflow as tf
import keras.models
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from keras.layers import GlobalMaxPooling2D, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from imutils import paths
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizer_experimental import adam
import glob
import cv2
import argparse
import random
import os

import aiml
import nltk
import numpy as np
import random
import string
import bs4 as bs
import urllib.request
import re


import nltk
nltk.download('punkt')
## The script under this will help respond to user queries 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

#####  Initialise NLTK Inference
from nltk.sem import Expression
from nltk.inference import ResolutionProver
read_expr = Expression.fromstring

#######################################################
#  Initialise Knowledgebase. 
#######################################################
import pandas




# import requests

# r = requests.get("http://www.google.com", 
#                  proxies={"http": "http://192.168.2.13:01"})
# print(r.text)

kb=[]
data = pandas.read_csv('kb.csv', header=None)
[kb.append(read_expr(row)) for row in data[0]]
# >>> ADD SOME CODES here for checking KB integrity (no contradiction), 
# otherwise show an error message and terminate

#######################################################

###Create kernel and learn files
responseAgent = 'aiml'
Kernel = aiml.Kernel()
Kernel.setTextEncoding(None)
Kernel.bootstrap(learnFiles="std-startup-B.xml")

# The code under this is for creating the corpus
# in order to get information i will use the wikipedia article on covid-19.
#The code retrieves data from the article then it is converted into lower case
#raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/COVID-19')

raw_html = urllib.request.urlopen('https://en.wikipedia.org/wiki/COVID-19')
raw_html = raw_html.read()

article_html = bs.BeautifulSoup(raw_html, 'lxml')

article_paragraphs = article_html.find_all('p')

article_text = ''

for para in article_paragraphs:
    article_text += para.text

article_text = article_text.lower()

#The code under removes all the empty spaces and any characters such as "'" from the text
article_text = re.sub(r'\[[0-9]*\]', ' ', article_text)
article_text = re.sub(r'\s+', ' ', article_text)

#This code divides the text into sentences for the cosine simlarity to work 
article_sentences = nltk.sent_tokenize(article_text)
article_words = nltk.word_tokenize(article_text)

#This is a helper function which will help edit the the user text for any punctuation. It also has the job of lemmatization the words
wnlemmatizer = nltk.stem.WordNetLemmatizer()

def perform_lemmatization(tokens):
    return [wnlemmatizer.lemmatize(token) for token in tokens]

punctuation_removal = dict((ord(punctuation), None) for punctuation in string.punctuation)

def get_processed_text(document):
    return perform_lemmatization(nltk.word_tokenize(document.lower().translate(punctuation_removal)))



def generate_response(user_input):
    covidbot_response = ''
    article_sentences.append(user_input)
## This changes the corpus into their vectorized form
    word_vectorizer = TfidfVectorizer(tokenizer=get_processed_text, stop_words='english')
    all_word_vectors = word_vectorizer.fit_transform(article_sentences)
# This function is used to find the cosine similarity
    similar_vector_values = cosine_similarity(all_word_vectors[-1], all_word_vectors)
    similar_sentence_number = similar_vector_values.argsort()[0][-2]

    matched_vector = similar_vector_values.flatten()
    matched_vector.sort()
    vector_matched = matched_vector[-2]
#If no similarity is found it will print this line
    if vector_matched == 0:
        covidbot_response = covidbot_response + "I could not understand you"
        return answer 

    else:
        covidbot_response = covidbot_response + article_sentences[similar_sentence_number]
        return covidbot_response
    
###################################

m = keras.models.load_model('geeksforgeeks.h5')
dirName = 'D:\Mine\Working\COVID_Image classification\data\pics';
def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles
# Get the list of all files in directory tree at given path
listOfFiles = getListOfFiles(dirName)
images = []
labels = []
myimages = []
valid_images =[".jpg", ".gif", ".png", ".tga", ".jpeg"]
for elem in listOfFiles:
    ext = os.path.splitext(elem)[1]
    if ext.lower() not in valid_images:
        continue
    #print(elem)        
    c = cv2.imread(elem)
    images.append(cv2.resize(c, [32, 32]))    
    myimages.append(c)

myLabels =['chest_x_ray', 'covid19_x_ray', 'Patient_lang_x_ray']    
images = np.array(images) / 255.0
###############################
        
        
# Welcome user
#######################################################
print("Hello, I am your friend CovidBot. You can ask me any question regarding Covid-19:")

#######################################################
# Main loop
#######################################################
rem=[]
while True:
    #get user input
    try:
        human_text = input('Human>')
        human_text = human_text.lower()
        answer = Kernel.respond(human_text)
    except (KeyboardInterrupt, EOFError) as e:
        print("Bye!")
        break
    #pre-process user input and determine response agent (if needed)
    responseAgent = 'aiml'
    #activate selected response agent
   
  
    if human_text != 'bye':
        if human_text == 'thanks' or human_text == 'thank you for your help' or human_text == 'thank you':
            continue_dialogue = False
            print("CovidBot: Your welcome")
    #post-process the answer for commands
    if len(answer) == 0:
        answer = generate_response(human_text)
    
    if answer[0] == '#':
        params = answer[1:].split('$')
        cmd = int(params[0])
        answer = Kernel.respond(human_text)
        if cmd == 0:
            print(params[1])
            break
        # >> YOU already had some other "if" blocks here from the previous 
        # courseworks which are not shown here.
        # Here are the processing of the new logical component:
        elif cmd == 31: # if input pattern is "I know that * is *"
            object,subject=params[1].split(' is ')
            expr1=read_expr(subject + '(' + object + ')')
            expr2=read_expr(subject + '(' + object + ')')
            
            if(ResolutionProver().prove(expr1, kb) == False and ResolutionProver().prove(expr2, kb) == False):
                kb.append(expr1)
                print('CovidBot: OK, I will remember that',object,'is', subject)
            else:
                print('CovidBot: ', object, 'is not 100%', subject)
            # >>> ADD SOME CODES HERE to make sure expr does not contradict 
            # with the KB before appending, otherwise show an error message.
 
            
        elif cmd == 32: # if the input pattern is "check that * is *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            if (ResolutionProver().prove(expr, kb) == True):
                print('Correct.')
            else:
               expr=read_expr(subject + '(' + object + ')')
               if (ResolutionProver().prove(expr, kb) == True): 
                   print('CovidBot: Incorrect')
               else:
                   print('Covidbot: I do not know')
                   

        elif cmd == 33: # if the input pattern is "check that * is not *"
            object,subject=params[1].split(' is ')
            expr=read_expr(subject + '(' + object + ')')
            if (ResolutionProver().prove(expr, kb) == False):
                print('Covidbot: Correct')
            else:
                print('CovidBot: incorrect')
        
        elif cmd == 34: # if the input pattern is "show me a *"
            pic_name = params[1] 
            while True:
                r = random.randint(0,len(images)-1)
                n = np.array(images[r])
                p = n.reshape(1, 32, 32, 3)
                predicted_label = myLabels[m.predict(p).argmax()]
                #print(predicted_label)
                if predicted_label.lower() == pic_name.lower():
                    plt.imshow(myimages[r])
                    rem = pic_name
                    plt.show()
                    break
        elif cmd == 35: # if the input pattern is "what is this x ray"
            print("This is " + rem)
        
                
            
                
                
        
                
                    
                
                
            
        
                
            
            
             
        elif cmd == 99:
            print("I did not get that, please try again.")
    else:
        print(answer)
            
    
       
        
       
        
      
     
   
    #