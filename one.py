# Importing the libraries
import numpy as np
import tensorflow as tf
import re
import time

########## PART 1 - DATA PREPROCESSING ##########

# Importing the dataset
lines = open('movie_lines.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')
conversations = open('movie_conversations.txt', encoding = 'utf-8', errors = 'ignore').read().split('\n')

# Creating a dictionary that maps each line and its id
id2line = {}
for line in lines:
    _line = line.split(' +++$+++ ')
    if len(_line) == 5:
        id2line[_line[0]] = _line[4]

# Creating a list of all of the conversations
conversations_ids = []
for conversation in conversations[:-1]:
    _conversation = conversation.split(' +++$+++ ')[-1][1:-1].replace("'", "").replace(" ", "")
    conversations_ids.append(_conversation.split(','))

# Getting separately the questions and the answers
questions = []
answers = []
for conversation in conversations_ids:
    for i in range(len(conversation) - 1):
        questions.append(id2line[conversation[i]])
        answers.append(id2line[conversation[i+1]])

# Doing a first cleaning of the texts
def clean_text(text):
    text = text.lower()
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "what is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"[-()\"#/@;:<>{}+=~|.?,]", "", text)
    return text

# Cleaning the questions
clean_questions = []
for question in questions:
    clean_questions.append(clean_text(question))

# Cleaning the answers
clean_answers = []
for answer in answers:
    clean_answers.append(clean_text(answer))

# Creating a dictionary that maps each word to its number of occurrences
word2count = {}
for question in clean_questions:
    for word in question.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1
for answer in clean_answers:
    for word in answer.split():
        if word not in word2count:
            word2count[word] = 1
        else:
            word2count[word] += 1

# Creating two dictionaries that map the questions words and the answers words to a unique integer
threshold_questions = 20
questionswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_questions:
        questionswords2int[word] = word_number
        word_number += 1
threshold_answers = 20
answerswords2int = {}
word_number = 0
for word, count in word2count.items():
    if count >= threshold_answers:
        answerswords2int[word] = word_number
        word_number += 1

# Adding the last tokens to these two dictionaries
tokens = ['<PAD>', '<EOS>', '<OUT>', '<SOS>']
for token in tokens:
    questionswords2int[token] = len(questionswords2int) + 1
for token in tokens:
    answerswords2int[token] = len(answerswords2int) + 1

# Creating the inverse dictionary of the answerswords2int dictionary
answersints2word = {w_i: w for w, w_i in answerswords2int.items()}

# Adding the End Of String token to the end of every answer
for i in range(len(clean_answers)):
    clean_answers[i] += ' <EOS>'

# Translating all the questions and the answers into integers
# and Replacing all the words that were filtered out by <OUT> 
questions_into_int = []
for question in clean_questions:
    ints = []
    for word in question.split():
        if word not in questionswords2int:
            ints.append(questionswords2int['<OUT>'])
        else:
            ints.append(questionswords2int[word])
    questions_into_int.append(ints)
answers_into_int = []
for answer in clean_answers:
    ints = []
    for word in answer.split():
        if word not in answerswords2int:
            ints.append(answerswords2int['<OUT>'])
        else:
            ints.append(answerswords2int[word])
    answers_into_int.append(ints)

# Sorting questions and answers by the length of questions
sorted_clean_questions = []
sorted_clean_answers = []
for length in range(1, 25 + 1):
    for i in enumerate(questions_into_int):
        if len(i[1]) == length:
            sorted_clean_questions.append(questions_into_int[i[0]])
            sorted_clean_answers.append(answers_into_int[i[0]])
            
            


########## PART 2 - BUILDING THE SEQ2SEQ MODEL ##########

#dealing with functions defining the architecture for the model

#in tensorflow all variables are in form of tensors which are kind of advanced arrays than numpy allowing computation in dl
#variables used in tensors are defined as TENSORFLOW PLACEHOLDERS
    #CREATING PLACEHODLERS FOR EACH VALUES- INPUT, TARGET, LEARNING RATE
    #IE WE are creatin these tensorflow PLACEHOLDER variables first to be used in training process later
# Creating placeholders for the inputs and the targets values
def model_inputs():
    #placeholder is a function here - with param( datatype, dim of matrix of input data ie sorted clean questions ie LIST of INTEGERS,name of the input)
    inputs = tf.placeholder(tf.int32, [None, None], name = 'input') 
    targets = tf.placeholder(tf.int32, [None, None], name = 'target')
    
    #hyperparamter learning rate
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    #dropout rate ie rate for overriding the neurons for iterations
    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')
    return inputs, targets, lr, keep_prob



#decoder accepts a certain format of targets hence preprocessing
    #target need to have special format
    #targets needs to be in BATCH and not single value
    #each sentence ie answer in the batch should start with <SOS> token ie putting <SOS> in start of each answer of each batch
    #HOW: Remove end of each answer in batch and the concatenate them
# Preprocessing the targets
    #word2int maps token to int to gget the END of LINE
    #batchsize ex=10
def preprocess_targets(targets, word2int, batch_size):
    #left contains left side ie start of each answer ie mATRIX of BATCHSIZE and 1st column
    #ARG: matrix to fill, values to fill
    #tf.fills the matrix(batchsize*1st column 1) by SOS tokens
    left_side = tf.fill([batch_size, 1], word2int['<SOS>'])
    
    #right = all answers in the batch except the last token
    #function extracts values from tensor specfied
    #strided.slice function(tensor to extract, starting point, end to where the extraction is to be doneie all columns except last one, cells to slide 1 cell by cell)
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    #precprocessedtarget values ready to be fed into the RNN
    #tf.concat([left and right side values], axis)
    #AXIS VALUES: horizontal=1 & vertical=0
    preprocessed_targets = tf.concat([left_side, right_side], 1)
    return preprocessed_targets


#ENCODER RNN LAYER

# Creating the Encoder RNN
    #ARGS: RNN INPUTS - model inputs, prepared inputs, inputquenstions, targets, LR
    # RNN SIZE - number of input tensors of encoder layer
    # NUM OF LAYERS
    # KEEP PROB for Dropout regularization for LSTM
    #SEQENCE_LENGHT - list of lenght of questions in each batch
def encoder_rnn(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    #creating lstm using TF ie it is an object of BASICLSTMCELL CLASS
    #ging through submodules to get BASICLSTMCELL class
    #ARG : number of input tensors
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    #apply dropout to lstm
    #apply dropoutwrapper class - WRAPS CREATED LSTM USING A DROPOUT VALUE
    #DROPOUT is deactivating certain neurons : typically 20%
    #ARGS: (lstm obj created, control dropoutrate value) 
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
    #ARG(LSTM WITH DROPOUT APPLIED times number of layers to be created in encoder RNN)
    #encoder cell consists of LSTM Cells
    encoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
    #encoder state
    #by bidirectional_dynamic_rnn returns 2 outputs: encoderstate and encoderoutput but we need only state value hence we place _ at first value
    #ARGS: (cellforwardRNN, cellbackwardRNN, listof length of each question in batch,modelinputs ie RNN inputs, datatype)
    encoder_output, encoder_state = tf.nn.bidirectional_dynamic_rnn(cell_fw = encoder_cell,
                                                       cell_bw = encoder_cell,
                                                       sequence_length = sequence_length,
                                                       inputs = rnn_inputs,
                                                       dtype = tf.float32)
    return encoder_state 


#DECODER RNN 
    #1. Decode training data
    #2. decode validation set
    #3. create decoder RNN

# Decoding the training set decodes observation of training set and return op of decoder
#ARGS: (encoderstate to be decoded, decodercell of RNN of decoder, embeddedinput ie mapping of words to vectors or integers,
    #seq lenght, decodingscope from variblescope class wrapping tf variables, op function, keepproab dropout regularization, batch size)
def decode_training_set(encoder_state, decoder_cell, decoder_embedded_input, sequence_length, decoding_scope, output_function, keep_prob, batch_size):
    #first initialzation needed, as 3D matrces containing zeros
    #ARGS ([matrix of size: lines ie batchsize, columns ie 1 ,elements in 3rd axis ie decodercelloutput size])
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    #prepare keys values and functions for attention
    #preprosess trainign data for attention
    
    #prepare attention args: attentionstate initialize previously step, attentionoption-bahdanu, number of units of decodercelloutputsize
    #attentionkeys: eys to be compared with target states
    #attentionvalues are values used to constrcut the context vectors which is retured by encoder which should be used by decoder as first element
    #attention_score_function computes similarity between keys and target states
    #attention_construct_function used to build the attention state
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    
    #training_decoder_function : decoding the training set
    #it can only decode only prepare_attention is done correctly
    # the attention features resulted from prepare_attention functiona re arguments of attentiondecodertrain function
    #ARGS: 
    # encoderstate value coming from from encoder_rnn_layer function
    #other args are attention features are from previous state & in the end specifying the name 
    # WE GOT ATTENTIONDECODER FUNCTION FR TRAINING OF RNN DECODER
    training_decoder_function = tf.contrib.seq2seq.attention_decoder_fn_train(encoder_state[0],
                                                                              attention_keys,
                                                                              attention_values,
                                                                              attention_score_function,
                                                                              attention_construct_function,
                                                                              name = "attn_dec_train")
    #GET DECODER OUTPUT:
    #OUTPUT: output, finalstate, finalcontextstate values are returned as OP by dynamic_rnn_decoder
    #ARGS: decodercell as arg of mother function, trainingdecoderfunction from prev step, decoderembeddedip as arg of mother function, lenghts, scope as decodingscope from args of mother function
    decoder_output, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                              training_decoder_function,
                                                                                                              decoder_embedded_input,
                                                                                                              sequence_length,
                                                                                                              scope = decoding_scope)
    #apply final dropout to decoder layer output
    # args: DECODEROUTPUT from prev step and keepprob param of dropout rate
    decoder_output_dropout = tf.nn.dropout(decoder_output, keep_prob)
    return output_function(decoder_output_dropout)



#2. New observation values ie decoding the testing data
    #also deals with validation data ie CROSS VALIDATION IS CARRIED OUT 10% is kept for cross validation
    #INFERENCE METHOD OF TF is used instead of TRAIN FUNCTION OF TF ie TEST RESULTS come out ie are inferred out of the weights stored


# Decoding the test/validation set
#ARGS: additional 4 ARGS: THESE are required for INFERERENCE FUNCTION
    # 1. SOS TOKEN ID ie start of string token id
    # 2. EOS TOKEN ID ie end of strng token id
    # 3. maximumlength - max length of an longest answer in batch
    # 4. num_words - total number of words of all answer - answord2int dict length 
def decode_test_set(encoder_state, decoder_cell, decoder_embeddings_matrix, sos_id, eos_id, maximum_length, num_words, decoding_scope, output_function, keep_prob, batch_size):
    # same required as previous function
    attention_states = tf.zeros([batch_size, 1, decoder_cell.output_size])
    # these params are needed and same as previous step
    attention_keys, attention_values, attention_score_function, attention_construct_function = tf.contrib.seq2seq.prepare_attention(attention_states, attention_option = "bahdanau", num_units = decoder_cell.output_size)
    
    #NEW FUNCTION
    #SAME ARGS AND NEW ONES and changes:
    # 1. outputfunction - not return the test pedictions but pass on
    # decoder_embeddings_matrix, sos_id, eos_id,maximum_length, num_words, from mother function args
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
    #final output isnt used to train but to get test results
    #just test_predictions values are used and not the other 2 values returned
    #ARGS: decoderccell is needed, not trainin but testdecoder function is used from previous step, scopre is needed for test and validation predictions
    test_predictions, decoder_final_state, decoder_final_context_state = tf.contrib.seq2seq.dynamic_rnn_decoder(decoder_cell,
                                                                                                                test_decoder_function,
                                                                                                                scope = decoding_scope)
    #no dropout is needed while testing
    #value coming from prev step
    return test_predictions

####################################################################################################################
#IMP:
'''
main layer of seq2seq model are encoder and decoder layer
'''    
####################################################################################################################

#CREATING DECODER_RNN IN ADDITION TO PREVIOUSLY MADE ENCODER_RNN

# Creating the Decoder RNN
#ARGS NEW: 
    # encoderstate - ouput of ENCODER and becomes IP of decoder
    # numwords- total number of words in answer corpus
    # numlayers- no of layers in out RNN
    # word2int ie answerword 2 int dictionary
    # dropout value - some training takes place in decoderRNN
    
def decoder_rnn(decoder_embedded_input, decoder_embeddings_matrix, encoder_state, num_words, sequence_length, rnn_size, num_layers, word2int, keep_prob, batch_size):
    #introducing scope which is the advanced value
    with tf.variable_scope("decoding") as decoding_scope:
        #creating LSTM layer - same as encoderRNN syntax
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        #applying dropout to the LSTM created in prev step with args the LSTM layer var, and keepprob param
        lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
        #multiRNN is used to stack several LSTM layers with dropout applied - coming same from encoderRNN with name change        
        decoder_cell = tf.contrib.rnn.MultiRNNCell([lstm_dropout] * num_layers)
        #initialize the weights associated to the neurons in connected layers of DECODERRNN
        #initilizerfunction is used - args: standard_deviation 
        weights = tf.truncated_normal_initializer(stddev = 0.1)
        #BIASES: initialized as zero values using zeros function from TF
        biases = tf.zeros_initializer()
        #classic python syntax function x(x is variable of function): values to return by function
        #VALUE RETURND - FULLY connected layers in the last layer of RNN
        #ARCH::: LSTM CELL->AT END we have fully connected LAYER
        #ARGS: function dealing with layers hence the layers mode=ule of TF
        # 1. inputs  = x
        # 2. no of op = no of words of answers
        # 3. activationfunction - default is relu
        # 4. normalizer - not needed hence NONE
        # 5. scope as decoding scope defined in previous step
        # 6. weights_initilizers which is TF arg in fully connected function - weights coming from previous steps
        # 7. biases_initializer TF arg - biases coming from prev steps
        # THIS WILL CREATE A LAST FULL CONNECTED LAYER GETTING THE FEATUERES FROM PREVIOUS STACKED LSTM LAYERS
        output_function = lambda x: tf.contrib.layers.fully_connected(x,
                                                                      num_words,
                                                                      None,
                                                                      scope = decoding_scope,
                                                                      weights_initializer = weights,
                                                                      biases_initializer = biases)
        #using the function we made to return the trainingpredictions
        #args are from the step we used to define the function
        training_predictions = decode_training_set(encoder_state,
                                                   decoder_cell,
                                                   decoder_embedded_input,
                                                   sequence_length,
                                                   decoding_scope,
                                                   output_function,
                                                   keep_prob,
                                                   batch_size)
        #take decoding scope and resuing the variables introduced in decoding scope
        decoding_scope.reuse_variables()
        #get test predictions ie getting answers from the chatbout after training
        #function decodetestset is coming from the function we defined previously
        #ARGS: 
            #1. encoderstate
            # decodercell, matrix
            # key indefifier os SOS 
            # key identifier of EOS
            # sequencelenght not including last token
            # totl num of words ie numwords
            # decoding scope as introduced in the beginning of mother function
            # output_function as defined in mother function
        test_predictions = decode_test_set(encoder_state,
                                           decoder_cell,
                                           decoder_embeddings_matrix,
                                           word2int['<SOS>'],
                                           word2int['<EOS>'],
                                           sequence_length - 1,
                                           num_words,
                                           decoding_scope,
                                           output_function,
                                           keep_prob,
                                           batch_size)
        #return the values returned from prev 2 steps
    return training_predictions, test_predictions

############################################################################################
# ASSEMBLE ENCODER RNN AND DECODER RNN
# final model which is used in chatiing while testing ie BRAIN 
# Building the seq2seq model
#ARGS: 
    # 1. Input - questions from dataset during training and actual questions asked during testing
    # 2. Target - answers being retrieved from dataset
    # 3. answernumwords - total words in all answers used previously in decoderRNN as numwords
    # 4. questionnumwords - total words in all questions used previously in decoderRNN as numwords
    # 5. encoder_embedding_size - no of dim of embedding matrix of encoder
    # 6. decoder_embedding_size - no of dim of embedding matrix of decoder
    # 7. rnn_size - used before
    # 8. num_layers - used in decoder RNN containing stacked LSTM
    # 9. questionswords2int - dict used to preprocess the targets
    # ALL PACKAGE TO RETURN training_predictions, test_predictions
def seq2seq_model(inputs, targets, keep_prob, batch_size, sequence_length, answers_num_words, questions_num_words, encoder_embedding_size, decoder_embedding_size, rnn_size, num_layers, questionswords2int):
    # DECODER TAKES ENCODER OP VALUES as input
    
    # used prev in encoderRNN
    # ARGS:
        # 1. inputs to be embedded
        # 2. totalwords of answer
        # 3. encoderembeddingsize - no of dim of embeddingmatrxi of encoder
        # 4. initilizer random - function from TF args- bounds of random numbers
    encoder_embedded_input = tf.contrib.layers.embed_sequence(inputs,
                                                              answers_num_words + 1,
                                                              encoder_embedding_size,
                                                              initializer = tf.random_uniform_initializer(0, 1))
    # encoderstate is the output of the ENCODER and it will be the input of the DECODER
    # using the ENCODERRNN function 
    # feeding the ENCODERRNN  THE ARGS: from the function defn we created
    encoder_state = encoder_rnn(encoder_embedded_input, rnn_size, num_layers, keep_prob, sequence_length)
    
    # preprocessed_targets are needed for training
    # using the function we created and corresponding args
    preprocessed_targets = preprocess_targets(targets, questionswords2int, batch_size)
    
    #decoder_embeddings_matrix is the matrxi of decoder which wil be used to get the decoder_embedded_input
    #created by using TF variable with  - dim of embedding matrix, 
    # the matrxi of filled with random number from uniform distribution using function, with lower and upper bound values to random - 0 to 1
    decoder_embeddings_matrix = tf.Variable(tf.random_uniform([questions_num_words + 1, decoder_embedding_size], 0, 1))
    #getting decoder_embedded_input using decoder_embeddings_matrix
    # args: decoder_embeddings_matrix, answers to questions ie preprocessed_targets
    decoder_embedded_input = tf.nn.embedding_lookup(decoder_embeddings_matrix, preprocessed_targets)
    
    # TRAININGPREDICTIONS AND TESTPREDICTIONS using DECODERRNN
    #ARGS: 
        # decoder_embedded_input & decoder_embeddings_matrix coming from prev step
    training_predictions, test_predictions = decoder_rnn(decoder_embedded_input,
                                                         decoder_embeddings_matrix,
                                                         encoder_state,
                                                         questions_num_words,
                                                         sequence_length,
                                                         rnn_size,
                                                         num_layers,
                                                         questionswords2int,
                                                         keep_prob,
                                                         batch_size)
    return training_predictions, test_predictions

##############################################################################################################################
    
########## PART 3 - TRAINING THE SEQ2SEQ MODEL ##########

# Setting the Hyperparameters
# EPOCHS: batches of ip into NN, forward propagating to encoder->to decoder RNN to get OP and backprop the loss back to tweak weights: one whole iteration
# EPOCHS MIN 50    
epochs = 100

# BATCHSIZE: try 128 or more to decrese training time
batch_size = 64

# RNN SIZE: 
rnn_size = 512

# NO of LAYERS IN ENCODER AND DECODER RNN INDIVIDUALLY
num_layers = 3

# ENCODING EMBEDDING SIZE: no of columns in embeddings matrix 
encoding_embedding_size = 512
decoding_embedding_size = 512

# LEARNING RATE: not too high or low,
learning_rate = 0.01

# % by which LR reduces over each iteration ie 90%
learning_rate_decay = 0.9

#min LR to handle the reduced LR by decay
min_learning_rate = 0.0001

# keep proab param : prob of neuron to be present during training- 50% is optimal
keep_probability = 0.5


# Defining a session
# session in which tensorflow training takes place
# InteractiveSession class object is needed to open the session
# reset the graph to be ready for training
tf.reset_default_graph()
session = tf.InteractiveSession()

