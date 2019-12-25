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

# Creating the encoder rnn
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                                    cell_bw = encoder_cell,
                                                                    sequence_length = sequence_length,
                                                                    inputs = rnn_inputs,
                                                                    dtype = tf.float32)
    return encoder_state

# Decoding the training set
def decode_training_set(encoder_state,decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option = 'bahdanau',
            num_units = decoder_cell.output_size)
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    decoder_output,decoder_final_state,decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                            training_decoder_function,
                                                                                                            decoder_embedded_input,
                                                                                                            sequence_length,
                                                                                                            scope = decoding_scope)
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)

# Decoding the Test/Validation Set
def decode_test_validation_set(encoder_state,decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(
            attention_states,
            attention_option = 'bahdanau',
            num_units = decoder_cell.output_size)
    test_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_inference(output_function,
                                                                              encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              decoder_embeddings_matrix,
                                                                              sos_id, 
                                                                              eos_id,
                                                                              maximum_length,
                                                                              num_words,
                                                                              name = "attn_dec_inf")
    test_predictions,decoder_final_state,decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              test_decoder_function,                                                                                                             
                                                                                                              scope = decoding_scope)
    return test_predictions

# Creating the decoder RNN
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, words_to_int, keep_prob, batch_size):
    with tf.variable_scope("decoding") as decoding_scope:
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm,input_keep_prob = keep_prob)
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        biases = tf.zeros_initializer()
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None, 
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        decoding_scope.reuse_variables()
        test_predictions =  decode_test_validation_set(encoder_state,
                                                       decoder_cell,
                                                       decoder_embeddings_matrix,
                                                       words_to_int['<SOS>'],
                                                       words_to_int['<EOS>'],
                                                       sequence_length - 1,
                                                       num_words,
                                                       sequence_length, # Was Missing from the Tutorial???
                                                       decoding_scope,
                                                       output_function,
                                                       keep_prob,
                                                       batch_size)
    return training_predictions, test_predictions

# Building the Seq2Seq Model
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questions_words_to_int):
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1, # +1 because Upper bound is excluded
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0,1))
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    preprocessed_targets = preprocess_targets(targets, questions_words_to_int, batch_size)
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    decoder_embedded_inputs = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_inputs,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questions_words_to_int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions
                                                         
# Setting the hyperparemeters
epochs = 100 # Number of full training being done to the LSTM
batch_size = 64 # Number of Inputs and Outputs being Processed at once
rnn_size = 512 # Number of Input Neurons
num_layers = 3 # Number of hidden layers
encoding_embedding_size = 512 # Number of columns in encoder embedding matrix
decoding_embedding_size = 512 # Number of columns in decoder embedding matrix
learning_rate = 0.01 # The amount of the error that is backpropagated after each batch
learning_rate_decay = 0.9 # Decays the learning rate to reduce overfitting
min_learning_rate = 0.0001 # Lower Limit of the learning rate so the decaying stops after a certain point
keep_probability = .5 # Must have the full name due to tensorflows api, the probability a neuron will be kept active during training, ie. the dropout rate

# Defining a session
tf.reset_default_graph() # This method should be called before each training session
session = tf.InteractiveSession() # Defines a tensorflow session

# Loading the model inputs
inputs, targets, lr, keep_prob = model_inputs()

# Setting the Sequence Length
sequence_length = tf.placeholder_with_default(25,None, name= 'sequence_length') #Using Placeholder with default for when the placeholder is not fed into the RNN (?)

# Getting the shape of the input tensor
input_shape = tf.shape(inputs)

# Getting the training and test predictions
training_predictions, test_predictions = seq2seq_model(tf.reverse(inputs, [-1]),
                                                       targets,
                                                       keep_prob,
                                                       batch_size,
                                                       sequence_length,
                                                       len(answers_words_to_int),
                                                       len(questions_words_to_int),
                                                       encoding_embedding_size,
                                                       decoding_embedding_size,
                                                       rnn_size,
                                                       num_layers,
                                                       questions_words_to_int)

# Setting up the Loss Error (the output error), the Optimizer, and Gradient Clipping (Keeps gradient between a max and a min value)
with tf.name_scope("optimization"):
    loss_error = tf.contrib.seq2seq.sequence_loss(training_predictions,
                                                  targets,
                                                  tf.ones([input_shape[0],sequence_length]))
    optimizer = tf.train.AdamOptimizer(learning_rate)
    gradients = optimizer.compute_gradients(loss_error)
    clipped_gradients = [(tf.clip_by_value(grad_tensor, -5.0, 5.0), grad_variable) for grad_tensor, grad_variable in gradients if grad_tensor is not None]
    optimizer_gradient_clipping = optimizer.apply_gradients(clipped_gradients)

# Padding the sequences with the <PAD> token
# Question: ['Who','are','you' , <PAD> , <PAD>, <PAD>, <PAD>]
# Answer:   [<SOS>, 'I', 'am', 'a', 'bot', '.' , <EOS>, <PAD>]
def apply_padding(batch_of_sequences, word_to_int):
    max_length = max( [len(sequence) for sequence in batch_of_sequences] ) # List Comprehension
    padded_sequence = [sequence + ([word_to_int['<PAD>']] * (max_length - len(sequence))) for sequence in batch_of_sequences] # List Comprehension
    return padded_sequence

# Splitting the data into batches of questions(inputs) and answers(targets)
def split_into_batches(questions, answers, batch_size):
    for batch_index in range(0, (len(questions) // batch_size)):
        start_index = batch_index * batch_size # The start index of the current batch
        questions_in_batch = questions[start_index:(start_index + batch_size)]
        answers_in_batch = answers[start_index:(start_index + batch_size)]
        padded_questions_in_batch = np.array(apply_padding(questions_in_batch, questions_words_to_int))
        padded_answers_in_batch = np.array(apply_padding(answers_in_batch, answers_words_to_int))
    yield padded_questions_in_batch, padded_answers_in_batch


# Splitting the questions and answers into the training and validation sets
training_validation_split = int(len(sorted_clean_questions) * 0.15) # Gets the index at 15% of the questions list
training_questions = sorted_clean_questions[training_validation_split:] # Test Set
training_answers = sorted_clean_answers[training_validation_split:] # Test Set
validation_questions = sorted_clean_questions[:training_validation_split] # Validation Set
validation_answers = sorted_clean_answers[:training_validation_split] # Validation Set  
    
# Training
    
batch_index_check_training_loss = 100  # Checks the training loss every 100 batches
batch_index_check_validation_loss = (len(training_questions) // batch_size// 2) - 1 # Checks the validation loss halfway and at the end of an epoch     
total_training_loss_error = 0 # Computes the training loss every 100 batches
list_validation_loss_error = [] # To use the early stopping technique to check to see if we have reached a loss error below the minimum of all loss errors
early_stopping_check = 0 # Increments every time we have failed to reduce the loss error
early_stopping_stop = 1000 # If we fail to decrease the error 1000 times the training will end
checkpoint = "chatbot_weights_ckpt" # File containing the saved weights so we do not have to retrain the chatbot
session.run(tf.global_variables_initializer())

############################################################### TRAINING LOOP ###############################################################

for epoch in range(1, epochs + 1):
    for batch_index, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(training_questions, training_answers, batch_size)):
        starting_time = time.time()
        _, batch_training_loss_error = session.run([optimizer_gradient_clipping, loss_error], {inputs: padded_questions_in_batch,
                                                                                               targets: padded_answers_in_batch,
                                                                                               lr: learning_rate,
                                                                                               sequence_length: padded_answers_in_batch.shape[1],
                                                                                               keep_prob: keep_probability})
        total_training_loss_error += batch_training_loss_error
        ending_time = time.time()
        batch_time = ending_time - starting_time
        if batch_index % batch_index_check_training_loss == 0:
            print('Epoch: {:>3}/{}, Batch: {:>4}/{}, Training Loss Error: {:>6.3f}, Training Time on 100 Batches: {:d} seconds'.format(epoch,
                                                                                                                                       epochs,
                                                                                                                                       batch_index,
                                                                                                                                       len(training_questions) // batch_size,
                                                                                                                                       int(total_training_loss_error / batch_index_check_training_loss),
                                                                                                                                       int(batch_time * batch_index_check_training_loss)))
                                                                                                                                        
        if batch_index % batch_index_check_validation_loss == 0 and batch_index > 0:
            total_validation_loss_error = 0
            starting_time = time.time()
            for batch_index_validation, (padded_questions_in_batch, padded_answers_in_batch) in enumerate(split_into_batches(validation_questions, validation_answers, batch_size)):
                batch_validation_loss_error = session.run(loss_error, {inputs: padded_questions_in_batch,
                                                                       targets: padded_answers_in_batch,
                                                                       lr: learning_rate,
                                                                       sequence_length: padded_answers_in_batch.shape[1],
                                                                       keep_prob: 1})
                total_validation_loss_error += batch_validation_loss_error
            ending_time = time.time()
            batch_time = ending_time - starting_time
            average_validation_loss_error = total_validation_loss_error / (len(validation_questions)/ batch_size)                                                                                                                      
            print('Validation Loss Error {:>6.3f}, Batch Validation Time: {:d} seconds'.format(average_validation_loss_error, int(batch_time)))
            learning_rate *= learning_rate_decay
            if learning_rate < min_learning_rate:
                learning_rate = min_learning_rate
            list_validation_loss_error.append(average_validation_loss_error)
            if average_validation_loss_error <= min(list_validation_loss_error):
                print('I speak better now!!')
                early_stopping_check = 0
                saver = tf.train.Saver()
                saver.save(session, checkpoint)
            else:
                print('Sorry, I do not speak better. I need to practice more.')
                early_stopping_check += 1
                if early_stopping_check == early_stopping_stop:
                    break
        if early_stopping_check == early_stopping_stop:
            print('My apologies, I cannot speak better anymore. This is the best I can do')
            break
print('Game Over')

############################################################### END TRAINING LOOP ###############################################################












