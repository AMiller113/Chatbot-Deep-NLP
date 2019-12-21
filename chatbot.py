# Building a chatbot with Deep NLP
import numpy as np
import tensorflow as tf
import re
import time

#Importing the dataset
lines = open('movie_lines.txt',encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open("movie_conversations.txt" ,encoding = 'utf-8', errors = 'ignore').read().split('\n')

#Creating a dictionary that matches the line IDs to the lines themselves
id_to_line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id_to_line[_line[0]] = _line[4]
        
#Create a list of all the conversations
conversations_ids = []
for conversations in conversations[:-1]:
    _conversations = conversations.split(' +++$+++ ')[-1][1:-1].replace("'" , "").replace(" ", "")
    conversations_ids.append(_conversations.split(','))

#Seperating the inputs and the outputs
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(0,len(conversation)-1,2):
        questions.append(id_to_line.get(conversation[i]))
        answers.append(id_to_line.get(conversation[i+1]))
        
#Doing a first cleaning of the text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm","i am", text)
    text = re.sub(r"he's","he is", text)
    text = re.sub(r"she's","she is", text)
    text = re.sub(r"that's","that is", text)
    text = re.sub(r"what's","what is", text)
    text = re.sub(r"where's","where is", text)
    text = re.sub(r"\'ll"," will" ,text)
    text = re.sub(r"\'ve"," have" ,text)
    text = re.sub(r"\'re"," are" ,text)
    text = re.sub(r"\'d"," would" ,text)
    text = re.sub(r"\'til","until" ,text)
    text = re.sub(r"\'bout","about" ,text)
    text = re.sub(r"won't","will not" ,text)
    text = re.sub(r"can't","can not" ,text)
    text = re.sub(r"[-()\"#/@{}+=~|.?,]","" ,text)
    return text

# Cleaning Questions and Answers
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))
    
# Creating a dictionary that maps each word to its number of occurences
word_occurences = {}
for question in clean_questions:
    for word in question.split():
        if word not in word_occurences:
            word_occurences[word] = 1
        else:
            word_occurences[word] += 1
                       
for answer in clean_answers:
    for word in answer.split():
        if word not in word_occurences:
            word_occurences[word] = 1
        else:
            word_occurences[word] += 1
            
# Mapping the questions and answers to a uniques integer
threshold = 20
questions_words_to_int = {}
word_number = 0
for word, count in word_occurences.items():
    if(count >= threshold):
        questions_words_to_int[word] = word_number
        word_number += 1

answers_words_to_int = {}
word_number = 0        
for word, count in word_occurences.items():
    if(count >= threshold):
        answers_words_to_int[word] = word_number
        word_number += 1    
        
# Adding the SOS,EOS, and OUT tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>' , '<OUT>' ,'<SOS>' ]
for token in tokens:
    questions_words_to_int[token] = len(questions_words_to_int) + 1

for token in tokens:
    answers_words_to_int[token] = len(answers_words_to_int) + 1

# Create an inverse dictionary of the answers words to int dictionary
answers_int_to_words = {w_i: w for w, w_i in answers_words_to_int.items()}

#Adding the EOS token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>' 

# Translating all questions and answers into integers
# Replace all words below the threshold with <OUT>
questions_as_ints = []
answers_as_ints = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questions_words_to_int:
            ints.append(questions_words_to_int.get('<OUT>'))
        else:
            ints.append(questions_words_to_int.get(word))
    questions_as_ints.append(ints)
            
for answer in clean_answers:
     ints = []
     for word in answer.split():
        if word not in answers_words_to_int:
            ints.append(questions_words_to_int.get('<OUT>'))
        else:
            ints.append(answers_words_to_int.get(word))
     answers_as_ints.append(ints)

#Sorting the questions and answers by the length of the quesitons
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_as_ints):
        if (len(i[1]) == length):
            sorted_clean_questions.append(questions_as_ints[i[0]])
            sorted_clean_answers.append(answers_as_ints[i[0]]) 
    
# Placeholders for the inputs and the targets
def model_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input')
    targets = tf.placeholder(tf.int32, [None, None], name = 'targets')
    learning_rate = tf.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, learning_rate, keep_prob

# Preprocessing the targets
def preprocess_targets(targets, words_to_ints, batch_size):
    left_side = tf.fill([batch_size,1],words_to_ints.get('<SOS>'))
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side,right_side], 1)
    return preprocessed_targets

# Creating the encoder
















































