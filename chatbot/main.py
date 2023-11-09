from multiprocessing import context
import re
from tabnanny import verbose
import nltk
import numpy as np
import random
import tflearn
import tensorflow
from tensorflow import keras
import json
import pickle
from nltk.stem.lancaster import LancasterStemmer
from django.conf import settings
import os

Stemmer = LancasterStemmer()

def getTraingAndOutputData(Mining = False):
  print('--------------- CALL ----------------')
  #dir = os.path.dirname(__file__)
  #filename = os.path.join(dir, 'intents.json')
  filename = 'D:\programs and projects\FAQ chatbot\chatbot\intents.json'

  with open(filename) as file:
      data = json.load(file)
  #filename = os.path.join(dir, "chat_bot_data.pickle")
  filename = 'D:\programs and projects\FAQ chatbot\chatbot\FAQ_chat_bot_data.pickle'
  if(Mining):
      WORDS = []
      LABELS = []
      DOCS = []
      PATTERNS_TAG = []

      for intent in data['intents']:
          for pattern in intent['patterns']:
              word = nltk.word_tokenize(pattern)
              WORDS.extend(word)
              DOCS.append(word)
              PATTERNS_TAG.append(intent['tag'])

          if("more" in intent.keys()):
              for more in intent['more']:
                for keys in intent['more'][more]:
                  DOCS.append(['what','is',intent['tag'],more,keys])
                  PATTERNS_TAG.append(intent['tag']+'_'+more+'_'+keys)
                  LABELS.append(intent['tag']+'_'+more+'_'+keys)

          if intent['tag'] not in LABELS:
              LABELS.append(intent['tag'])

      WORDS = sorted(list(set([Stemmer.stem(word.lower()) for word in WORDS if word != '?'])))
      LABELS = sorted(LABELS)

      print(DOCS)
      print(PATTERNS_TAG)

      training = []
      output = []
      out_empty = [0 for _ in range(len(LABELS))]

      for count,doc in enumerate(DOCS):
          bag = []
          words = [Stemmer.stem(word) for word in doc]
          
          for word in WORDS:
              if word in words:
                  bag.append(1)
              else:
                  bag.append(0)
          
          output_row = out_empty[:]
          output_row[LABELS.index(PATTERNS_TAG[count])] = 1
          training.append(bag)
          output.append(output_row)
      training = np.array(training)
      output = np.array(output)
      with open(filename, 'wb') as file:
        pickle.dump((WORDS,LABELS,training,output),file)
  else:
      with open(filename, 'rb') as file:
        print('File ------------------------> ',file)
        WORDS,LABELS,training,output = pickle.load(file)
  return WORDS,LABELS,training,output,data


def DNN_Model(training,output,Train_Model = False):

  tensorflow.compat.v1.reset_default_graph()

  net = tflearn.input_data(shape = [None, len(training[0])])
  net = tflearn.fully_connected(net, 8, activation = "relu")
  net = tflearn.fully_connected(net, len(output[0]), activation = "softmax")
  net = tflearn.regression(net)

  model = tflearn.DNN(net)
  #filename = os.fspath("chatbot\chat_model.tflearn")
  filename = "D:\programs and projects\FAQ chatbot\chatbot\FAQ_chat_bot_model.tflearn"
  if(Train_Model):
    model.fit(training,output,n_epoch = 100,batch_size = 8, show_metric = True)
    model.save(filename)
  else:
    model.load(filename) 
  return model

  
WORDS, LABELS, training, output, data = getTraingAndOutputData(False)

model = DNN_Model(training,output,Train_Model = False)
contexts = ""

def finalPredict(string):
  global contexts
  messages = []
  results = model.predict([bagOfWords(string, WORDS)])
  results_index = np.argmax(results)
  resultTag = LABELS[results_index]
  intent = findIntent(resultTag)
  print(resultTag)
  if(intent == None):
    keys = resultTag.split("_")
    intent = findIntent(keys[0])
    return intent['more'][keys[1]][keys[2]]
  else: 
    finalResponses = intent['responses']
  messages.append(random.choice(finalResponses))
  if(intent['tag'] == 'confirmations_positive' and contexts != ""):
    intent = findIntent(contexts)
    finalResponses = getResponses(intent)
    messages.append(random.choice(finalResponses)) 
  else:
    contexts = intent['context_set']
    if(contexts != ''):
        intent = findIntent(contexts)
        finalResponses = getQuestions(intent)
        messages.append(random.choice(finalResponses))
  return messages

def chat():
  print("Start talking with the bot (type quite to stop)!")
  while True:
    string = input("You: ")
    if(string.lower() == 'quit'):
      break
    print('BOT',finalPredict(string))

def findIntent(tag):
  for intent in data['intents']:
      if intent['tag'] == tag:
        return intent
  else: 
    return None

def getResponses(intent):
  return intent['responses']

def getQuestions(intent):
  return intent['questions']

def bagOfWords(sentence, dictionary):

  bag = []
  words = nltk.word_tokenize(sentence)
  words = [Stemmer.stem(word.lower()) for word in words]
  for dic in dictionary:
      if dic in words:
        bag.append(1)
      else:
        bag.append(0)
  return np.array(bag)

#chat()