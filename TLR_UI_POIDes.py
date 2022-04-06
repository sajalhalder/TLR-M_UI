"""
Research Article: POI Recommendation with Queuing Time and User Interest Awareness (In Submission DMKD)
Authors: Sajal Halder, Jeffrey Chan, Xiuzhen Zhang and Kwan Hui Lim
Implemented By: Sajal Halder, PhD Candidate, RMIT University, Australia
Implementation Time: November 2022 - May 2021
Description: POI description based users interest in top-k POI recommendation
"""
import tensorflow as tf
from konlpy.tag import Twitter
import numpy as np
import os

from sklearn.model_selection import train_test_split

import pandas as pd
import re
import random
from tqdm import tqdm
import timeit

import shutil
import math
import warnings
import Doc2Vec as d2v
warnings.filterwarnings('ignore')
#
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)

FILTERS = "([~.,!?\"':;)(])"
PAD = "<PADDING>"
STD = "<START>"
END = "<END>"
UNK = "<UNKNWON>"

PAD_INDEX = 0
STD_INDEX = 1
END_INDEX = 2
UNK_INDEX = 3

MARKER = [PAD,STD, END, UNK]
CHANGE_FILTER = re.compile(FILTERS)

USERID = {}

from configs import DEFINES
DATA_OUT_PATH = './data_out/'




POPULARITY = {}


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    #Positional Encoding
    # paper: https://arxiv.org/abs/1706.03762
    # P E(pos,2i) = sin(pos/100002i/dmodel)
    # P E(pos,2i+1) = cos(pos/100002i/dmodel)
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)

def layer_norm(inputs, eps=1e-6):
    # LayerNorm(x + Sublayer(x))
    feature_shape = inputs.get_shape()[-1:]
    # Pass the mean and standard deviation.
    mean = tf.keras.backend.mean(inputs, [-1], keepdims=True)
    std = tf.keras.backend.std(inputs, [-1], keepdims=True)
    beta = tf.Variable(tf.zeros(feature_shape), trainable=False)
    gamma = tf.Variable(tf.ones(feature_shape), trainable=False)

    return gamma * (inputs - mean) / (std + eps) + beta


def sublayer_connection(inputs, sublayer):
    # LayerNorm(x + Sublayer(x))
    return tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(layer_norm(inputs + sublayer))


def feed_forward(inputs, num_units):
    # FFN(x) = max(0, xW1 + b1)W2 + b2
    with tf.compat.v1.variable_scope("feed_forward", reuse=tf.compat.v1.AUTO_REUSE):
        outputs = tf.keras.layers.Dense(num_units[0], activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(outputs)
        return tf.keras.layers.Dense(num_units[1])(outputs)


def scaled_dot_product_attention(query, key, value, masked=False):
    #Attention(Q, K, V ) = softmax(QKt / root dk)V
    key_seq_length = float(key.get_shape().as_list()[-2])
    key = tf.transpose(key, perm=[0, 2, 1])
    outputs = tf.matmul(query, key) / tf.sqrt(key_seq_length)

    if masked:
        diag_vals = tf.ones_like(outputs[0, :, :])
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
        masks = tf.compat.v1.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])

        paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

    attention_map = tf.nn.softmax(outputs)

    return tf.matmul(attention_map, value)


def multi_head_attention(query, key, value, heads, masked=False):
    # MultiHead(Q, K, V ) = Concat(head1, ..., headh)WO
    with tf.compat.v1.variable_scope("multi_head_attention", reuse=tf.compat.v1.AUTO_REUSE):
        feature_dim = query.get_shape().as_list()[-1]

        query = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(query)
        key = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(key)
        value = tf.keras.layers.Dense(feature_dim, activation=tf.nn.relu)(value)

        query = tf.concat(tf.split(query, heads, axis=-1), axis=0)
        key = tf.concat(tf.split(key, heads, axis=-1), axis=0)
        value = tf.concat(tf.split(value, heads, axis=-1), axis=0)

        attention_map = scaled_dot_product_attention(query, key, value, masked)

        attn_outputs = tf.concat(tf.split(attention_map, heads, axis=0), axis=-1)

        return attn_outputs


def conv_1d_layer(inputs, num_units):
    # Another way of describing this is as two convolutions with kernel size 1
    with tf.variable_scope("conv_1d_layer", reuse=tf.AUTO_REUSE):
        outputs = tf.keras.layers.Conv1D(num_units[0], kernel_size = 1, activation=tf.nn.relu)(inputs)
        outputs = tf.keras.layers.Dropout(rate=DEFINES.dropout_width)(outputs)
        return tf.keras.layers.Conv1D(num_units[1], kernel_size = 1)(outputs)


def encoder_module(inputs, num_units, heads):
    self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs, heads))

    if DEFINES.conv_1d_layer:
        network_layer = conv_1d_layer(self_attn, num_units)
    else:
        network_layer = feed_forward(self_attn, num_units)

    outputs = sublayer_connection(self_attn, network_layer)
    return outputs


def decoder_module(inputs, encoder_outputs, num_units, heads):
    # sublayer_connection Parameter input Self-Attention
    # multi_head_attention parameter Query Key Value Head masked
    masked_self_attn = sublayer_connection(inputs, multi_head_attention(inputs, inputs, inputs, heads, masked=True))

    self_attn = sublayer_connection(masked_self_attn, multi_head_attention(masked_self_attn, encoder_outputs, encoder_outputs, heads))

    if DEFINES.conv_1d_layer:
        network_layer = conv_1d_layer(self_attn, num_units)
    else:
        network_layer = feed_forward(self_attn, num_units)

    outputs = sublayer_connection(self_attn, network_layer)
    return outputs



def encoder(inputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = encoder_module(outputs, num_units, heads)

    return outputs


def decoder(inputs, encoder_outputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = decoder_module(outputs, encoder_outputs, num_units, heads)

    return outputs


def find_idcg():
    idcg_5, idcg_10 = 0.0, 0.0

    for i in range(5):
        idcg_5 = idcg_5 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    for i in range(10):
        idcg_10 = idcg_10 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    return idcg_5, idcg_10



def largest_indices(array: np.ndarray, n: int) -> tuple:
    """Returns the n largest indices from a numpy array.
    Arguments:
        array {np.ndarray} -- data array
        n {int} -- number of elements to select
    Returns:
        tuple[np.ndarray, np.ndarray] -- tuple of ndarray
        each ndarray is index
    """
    # array[0] = -10000.00  # Avoid first index for selection

    flat = array.flatten()
    indices = np.argpartition(flat, -n)[-n:]

    indices = indices[np.argsort(-flat[indices])]

    values = np.sort(np.array(array))[::-1]

    return np.asarray(np.unravel_index(indices, array.shape))[0], values[0:10]



def Model(features, labels, mode, params):
    TRAIN = mode == tf.estimator.ModeKeys.TRAIN
    EVAL = mode == tf.estimator.ModeKeys.EVAL
    PREDICT = mode == tf.estimator.ModeKeys.PREDICT

    positional_encoded = positional_encoding(params['embedding_size'], DEFINES.max_sequence_length)
    positional_encoded.trainable = True
    if TRAIN:
        position_inputs = tf.tile(tf.range(0, DEFINES.max_sequence_length), [DEFINES.batch_size])
        position_inputs = tf.reshape(position_inputs, [DEFINES.batch_size, DEFINES.max_sequence_length])
    else:
        position_inputs = tf.tile(tf.range(0, DEFINES.max_sequence_length), [1])
        position_inputs = tf.reshape(position_inputs, [1, DEFINES.max_sequence_length])

    embedding = tf.compat.v1.get_variable(name='embedding', shape=[params['vocabulary_length'], params['embedding_size']],dtype=tf.float32) #, initializer=tf.compat.layers.xavier_initializer())
    W = tf.compat.v1.tile(tf.expand_dims(tf.compat.v1.get_variable('W', [1, params['embedding_size']], dtype=tf.float32), 0),[DEFINES.batch_size, 1, 1])
    embedding_user1 = tf.compat.v1.get_variable("embedding_user", shape = [params['user_length'], params['embedding_size']], dtype=tf.float32) # initializer=tf.contrib.layers.xavier_initializer())
    W_u = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_u", [params['embedding_size'], params['embedding_size']], dtype=tf.float32),0), [DEFINES.batch_size, 1, 1])

    W_p = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_p", [params['embedding_size'], params['embedding_size']], dtype= tf.float32), 0), [DEFINES.batch_size, 1, 1])


    encoder_inputs = tf.matmul(tf.nn.embedding_lookup(embedding, features['input']),W_p)
    decoder_inputs = tf.matmul(tf.nn.embedding_lookup(embedding, features['output']),W_p)
    encoder_features = tf.matmul(tf.expand_dims(tf.cast(features['in_distance'], tf.float32), 2), W)  + tf.matmul(tf.expand_dims(tf.cast(features['in_time'], tf.float32), 2), W)
    encoder_UI_features = tf.matmul(tf.expand_dims(tf.cast(features['in_poisim'], tf.float32), 2), W) #+ tf.matmul(tf.expand_dims(tf.cast(features['in_cate'], tf.float32), 2), W)

    encoder_user_features = tf.matmul(tf.nn.embedding_lookup(embedding_user1, features['in_users']),W_u) # tf.matmul(tf.expand_dims(tf.cast(features['in_users'], tf.float32), 2), W)

    encoder_inputs  += encoder_features + encoder_user_features + encoder_UI_features


    position_encode = tf.nn.embedding_lookup(positional_encoded, position_inputs)

    encoder_inputs = encoder_inputs + position_encode
    decoder_inputs = decoder_inputs + position_encode

    # dmodel = 512, inner-layer has dimensionality df f = 2048.  (512 * 4)
    # dmodel = 128 , inner-layer has dimensionality df f = 512  (128 * 4)
    # H = 8 N = 6
    # H = 4 N = 2
    encoder_outputs = encoder(encoder_inputs,
                              [params['hidden_size'] * 4, params['hidden_size']], DEFINES.heads_size,
                              DEFINES.layers_size)
    decoder_outputs = decoder(decoder_inputs,
                              encoder_outputs,
                              [params['hidden_size'] * 4, params['hidden_size']], DEFINES.heads_size,
                              DEFINES.layers_size)

    logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs)




    if PREDICT:
        predictions = {
            'topk': tf.nn.top_k(logits[:, 0:1, :], 10)[1],  # tf.nn.top_k(logits[:,0:1,:],10)[1]
            'logit': logits,
            'reward': tf.nn.top_k(logits[:, 0:1, :], 10)[0]

        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
    if DEFINES.mask_loss:
        embedding_tile = tf.tile(tf.expand_dims(embedding, 0), [DEFINES.batch_size, 1, 1])
        linear_outputs = tf.matmul(decoder_outputs, embedding_tile, transpose_b = True)

        #mask_zero = 1 - tf.cast(tf.equal(labels, 0),dtype=tf.float32)
        mask_end = 1 - tf.cast(tf.equal(labels, 2), dtype=tf.float32)
        labels_one_hot = tf.one_hot(indices = labels, depth = params['vocabulary_length'], dtype = tf.float32) # [BS, senxlen, vocab_size]
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = labels_one_hot, logits = linear_outputs)
        #loss = loss * mask_zero
        loss = loss * mask_end
        loss = tf.reduce_mean(loss)
    else:

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))


    if EVAL:
        correct_prediction_5 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.tile(labels[:,0:1],[1,5]), tf.int32), tf.nn.top_k(logits[:,0:1,:],5)[1]), tf.float32))*5 #/ tf.cast((tf.shape(logits)[0] * tf.shape(logits)[1]),tf.float32) #DEFINES.batch_size * params['vocabulary_length'])
        correct_prediction_10 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.tile(labels[:, 0:1], [1, 10]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 10)[1]),tf.float32)) *10
        recall_5 = (correct_prediction_5, correct_prediction_5)
        precision_5 = (correct_prediction_5/5, correct_prediction_5/5)
        recall_10 = (correct_prediction_10, correct_prediction_10)
        precision_10 = (correct_prediction_10/10, correct_prediction_10/10)
        f1_5 = (2*recall_5[0]*precision_5[0] /(recall_5[0] + precision_5[0] + 1e-8), 2*recall_5[0]*precision_5[0] /(recall_5[0] + precision_5[0] + 1e-8))
        f1_10 = (2 * recall_10[0]* precision_10[0] / (recall_10[0] + precision_10[0] + 1e-8), 2 * recall_10[0]* precision_10[0] / (recall_10[0] + precision_10[0] + 1e-8))


        idcg_5, idcg_10 = find_idcg()
        ndcg_5 = tf.reduce_mean(tf.math.log(2.0) / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(tf.cast(tf.tile(labels[:, 0:1], [1, 1]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 5)[1]), tf.int64)),tf.float32) + 2.0)  ))  / idcg_5 #* tf.cast(DEFINES.batch_size, tf.float32))#
        ndcg_10 = tf.reduce_mean(tf.math.log(2.0) / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(tf.cast(tf.tile(labels[:, 0:1], [1, 1]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 10)[1]), tf.int64)),tf.float32) + 2.0) ) )/ idcg_10 #* tf.cast(DEFINES.batch_size, tf.float32))

        ndcg_5 = (ndcg_5, ndcg_5)
        ndcg_10 = (ndcg_10, ndcg_10)



        metrics = {'recall_5': recall_5, 'precision_5': precision_5, 'f1_5':f1_5, 'recall_10': recall_10, 'precision_10': precision_10, 'f1_10':f1_10,'ndcg_5': ndcg_5, 'ndcg_10': ndcg_10} #, 'pop_5':pop5, 'pop10':pop10}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


"""
    Clear all existing build in files including folders and subfolders 
"""
#**********************************************************************************************************************
def clearExistingFile():
    with os.scandir(DEFINES.check_point_path) as entries:
        for entry in entries:
            if entry.is_file() or entry.is_symlink():
                os.remove(entry.path)
            elif entry.is_dir():
                shutil.rmtree(entry.path)
#**********************************************************************************************************************


""" Time Function: Return times sequence """
#**********************************************************************************************************************
# sequence based time distance
def time_function(dfVisits, train_pois):

    train_time = []
    train_time.extend([1])  # initially 100 second for first POI
    dfVisit1 = dfVisits.sort_values('takenUnix', ascending=True).drop_duplicates('poiID').reset_index()

    for i in range(1, len(train_pois)):
        t = (dfVisit1[dfVisit1.poiID == train_pois[i]].takenUnix.values[0] - dfVisit1[dfVisit1.poiID == train_pois[i-1]].takenUnix.values[0] + 1)//60

        train_time += [t]

    return train_time
#**********************************************************************************************************************


def to_frequency_table(data):
    frequencytable = {}
    for key in data:
        if 'POI_'+ str(key) in frequencytable:
            frequencytable['POI_'+str(key)] += 1
        else:
            frequencytable['POI_'+str(key)] = 1
    return frequencytable


"""
    Preproces data based on datasets. 
    Both min_poi and min_user are 3
    Constract based on sequence ID 
"""
#**********************************************************************************************************************
def preprocess(dataset):
    min_poi = 20 if dataset in "Gowalla,Foursquare" else 3
    min_user = 20 if dataset in "Gowalla,Foursquare" else 3



    dfVisits = pd.read_excel('DataExcelFormat/userVisits-' + dataset + '-allPOI.xlsx')

    dfVisits = dfVisits[dfVisits.takenUnix > 0]
    dfVisits['user_freq'] = dfVisits.groupby('nsid')['nsid'].transform('count')
    dfVisits['poi_freq'] = dfVisits.groupby('poiID')['poiID'].transform('count')

    dfVisits = dfVisits[dfVisits.user_freq >= min_user]
    dfVisits = dfVisits[dfVisits.poi_freq >= min_poi]

    dfVisits = dfVisits[['nsid', 'poiID', 'takenUnix', 'seqID']]

    poiNumber = min(max(dfVisits.poiID), DEFINES.POI_numbers)   # Find datasets based on POI_numbers

    dfVisits = dfVisits[dfVisits.poiID < poiNumber]


    df = pd.DataFrame(columns=['Q', 'A', 'U','T'])
    # df_test = pd.DataFrame(columns=['Q', 'A', 'U', 'T'])
    train_time = []


    train_part = 0.7
    window_size = DEFINES.max_sequence_length
    sequences = dfVisits.seqID.unique()
    # print(sequences)
    max_len = 0
    for seq in sequences:
        tempdfVisits = dfVisits[dfVisits.seqID == seq]
        tempdfVisits = tempdfVisits.sort_values(['takenUnix'], ascending=[True])
        user = tempdfVisits.iloc[0].nsid
        pois = tempdfVisits.poiID.unique()
        max_len = max(len(pois)*train_part+0.5, len(pois)*(1-train_part)+ 0.5) if max_len < max(len(pois)*train_part+0.5, len(pois)*(1-train_part)+ 0.5) else max_len

        pois = list(pois)
        # Update user interest
        if user not in USERID:
            USERID[user] = len(USERID)

        userid = USERID[user]

        time_seq = time_function(tempdfVisits, pois)
        max_len = len(pois)-1 if max_len < len(pois)-1 else max_len
        for i in range(len(pois) - 2):

            startW = max(i - window_size, 0)
            endW = i + 3  #
            train_d = ['POI_' + str(poi) for poi in pois[startW:endW-1]]
            test_d = ['POI_' + str(poi) for poi in pois[endW-1:endW]]
            time = [t for t in time_seq[startW:endW-1]]
            time[0] = 1
            userseq = [int(userid) for i in range(len(train_d))]
            df.at[df.shape[0]] = (train_d, test_d, userseq, time)
            # if i < (len(pois)-2)*train_part:
            #     df.at[df.shape[0]] = (train_d, test_d, userseq, time )
            # else:
            #     df_test.at[df_test.shape[0]] = (train_d, test_d, userseq, time)

            train_time.append(time)


    characters = ["'", ",", "[", "]"]
    for char in characters:
        df['Q'] = df['Q'].apply(str).str.replace(char, '')
        df['A'] = df['A'].apply(str).str.replace(char, '')
        df['U'] = df['U'].apply(str).str.replace(char, '')
        df['T'] = df['T'].apply(str).str.replace(char, '')



    #Save input sequence
    df.to_csv('data_in/input_'+dataset+'_train.csv', index=False)

    return max_len, train_time
#**********************************************************************************************************************

"""
Find Coordinates values and categoreis values. 
"""
#**********************************************************************************************************************
def findCordinates_Category(dataset):
    Cordinats = {}
    Category = {}
    dfNodes = pd.read_excel('DataExcelFormat/POI-' + dataset + '.xlsx')
    for i in range(len(dfNodes)):
        poiID = 'POI_'+ str(dfNodes.iloc[i].poiID)
        lati = dfNodes.iloc[i].lat
        long = dfNodes.iloc[i].long
        Cordinats[poiID] = (lati, long)
        for j in range(len(dfNodes)):
            try:
                if dfNodes.iloc[i].theme == dfNodes.iloc[j].theme:
                    Category['POI_'+str(dfNodes.iloc[i].poiID), 'POI_'+str(dfNodes.iloc[j].poiID)] = 1
                else:
                    Category['POI_'+str(dfNodes.iloc[i].poiID), 'POI_'+str(dfNodes.iloc[j].poiID)] = 0
            except:
                Category['POI_' + str(dfNodes.iloc[i].poiID), 'POI_' + str(dfNodes.iloc[j].poiID)] = 0
                pass

    return Cordinats, Category
#**********************************************************************************************************************


def prepro_like_morphlized(data):
    # Morpheme analysis module object
    # Create.

    morph_analyzer = Twitter()
    # Receive morpheme TalkNise result sentences
    # Create a list.
    result_data = list()
    # Tokenize every sentence in the data
    # Declare a loop to do it.
    for seq in tqdm(data):
        # Talked through Twitter.morphs function
        # Get a list object and again based on whitespace
        # To reconstruct the string.
        morphlized_seq = " ".join(morph_analyzer.morphs(seq.replace(' ', '')))
        result_data.append(morphlized_seq)

    return result_data


def data_tokenizer(data):
    # Create an array to talk to
    words = []
    for sentence in data:
        # FILTERS = "([~.,!? \" ':;) (]) "
        # Regularize expressions for values like the above filter
        # This is the part that converts  all to   "" through  # .
        sentence = re.sub(CHANGE_FILTER, "", sentence)
        for word in sentence.split():
            words.append(word)
    # Created through toggining and regular expression
    # Pass values.
    return [word for word in words if word]


def make_vocabulary(vocabulary_list):
    # List a key with words and values with indexes
    # Make a dictionary.
    char2idx = {char: idx for idx, char in enumerate(vocabulary_list)}
    # List a key with an index and a value with a word
    # Make a dictionary.
    idx2char = {idx: char for idx, char in enumerate(vocabulary_list)}
    # Hand over two dictionaries.
    return char2idx, idx2char




"""Load Vocabulary"""
def load_vocabulary(dataset):
    # Prepare an array to hold the dictionary
    vocabulary_list = []
    # After configuring the dictionary, proceed to save it as a file.
    # Check the existence of the file.
    dataPath = "data_in/input_"+dataset+"_train.csv"
    if (os.path.exists(dataPath)):
        # Through the judgment because data exists
        # Load data
        data_df = pd.read_csv(dataPath, encoding='utf-8')
        # print("data_df = ", data_df)

        question, answer = list(data_df['Q']), list(data_df['A'])
        if DEFINES.tokenize_as_morph:  # Tokenizer processing by morpheme
            question = prepro_like_morphlized(question)
            answer = prepro_like_morphlized(answer)

        data = []
        # Extend questions and answers
        # Make an unstructured array through.
        data.extend(question)
        data.extend(answer)
        # This is the part of tokenizer processing
        words = data_tokenizer(data)

        words = list(set(words))

        PAD = "<PADDING>"
        STD = "<START>"
        END = "<END>"
        UNK = "<UNKNWON>"
        words[:0] = MARKER


        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:

                vocabulary_file.write(word + '\n')


    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    char2idx, idx2char = make_vocabulary(vocabulary_list)

    return char2idx, idx2char, len(char2idx)


"""Load Data"""


def load_data(dataset,time):

    data_df = pd.read_csv('data_in/input_' + dataset + '_train.csv', header=0)
    # data_df_test = pd.read_csv('data_in/input_' + dataset + '_test.csv', header=0)
    # train_input, train_label, train_user, train_time = list(data_df['Q']), list(data_df['A']), list(data_df['U']), list(data_df['T'])
    # eval_input, eval_label, eval_user, eval_time = list(data_df_test['Q']), list(data_df_test['A']), list(data_df_test['U']), list(
    #     data_df['T'])

    question, answer, user, timeid = list(data_df['Q']), list(data_df['A']), list(data_df['U']), list(data_df['T'])
    random_seed = random.randint(10,100)
    #print(random_seed)
    train_input, eval_input1, train_label, eval_label1 = train_test_split(question, answer, test_size=0.30 , random_state=random_seed)
    train_user, eval_user, train_timeid, eval_timeid = train_test_split(user, timeid, test_size=0.30, random_state= random_seed)
    # devide time based on test size
    train_time, eval_time = train_test_split(time, test_size = 0.30, random_state= random_seed)
    eval_input, test_input, eval_label, test_label = train_test_split(eval_input1, eval_label1, test_size=0.66, random_state=random_seed+1)



    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    #Sort train_input based on sequence length
    sorted_index = len_argsort(train_input)
    train_input = [train_input[i] for i in sorted_index]
    train_label = [train_label[i] for i in sorted_index]
    train_time = [train_time[i] for i in sorted_index]
    train_user = [train_user[i] for i in sorted_index]
    train_timeid = [train_timeid[i] for i in sorted_index]

    # Sort eval_input based on sequence length
    sorted_index = len_argsort(eval_input1)
    eval_input1 = [eval_input1[i] for i in sorted_index]
    eval_label1 = [eval_label1[i] for i in sorted_index]
    eval_time = [eval_time[i] for i in sorted_index]
    eval_user = [eval_user[i] for i in sorted_index]
    eval_timeid = [eval_timeid[i] for i in sorted_index]

    user = (train_user, eval_user)
    timeid = (train_timeid, eval_timeid)

    return train_input, train_label, eval_input1, eval_label1, test_input, test_label, train_time, eval_time, user, timeid


"""Enc processing """

# The value and key to be indexed are words
# Take a dictionary whose value is an index.
def enc_processing(value, users, dictionary, cordinates):
    # Holding index values
    #print("Value = ", value)
    # Array (cumulative)
    sequences_input_index = []
    sequence_input_distance = []
    sequence_input_user = []

    # One encoded sentence
    # Has a length (accumulates)
    sequences_length = []
    # Whether to use morpheme toning
    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)
    #print("Value = ", value)
    # Blows line by line.

    for user in users:
        sequence_user = []
        for u in user.split(" "):
            sequence_user.append(int(u))
        if len(sequence_user) > DEFINES.max_sequence_length:
            sequence_user = sequence_user[:DEFINES.max_sequence_length]
        sequence_user += (DEFINES.max_sequence_length - len(sequence_user)) * [dictionary[PAD]]
        sequence_input_user.append(sequence_user)

    for sequence in value:
        # FILTERS = "([~.,!? \" ':;) (]) "
        # Using normalization, the filter contains
        # Replace values ​​with "".
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # When encoding a sentence
        # This is an array to have.
        sequence_index = []
        sequence_distance = []

        pre_word = -1

        for word in sequence.split():
            # Report if truncated words exist in the dictionary
            # Get the value and add it to sequence_index.
            if dictionary.get(word) is not None:
                sequence_index.extend([dictionary[word]])
                value = euclidean_dist(cordinates[word], cordinates[pre_word]) if cordinates.get(pre_word) is not None else 1.0
                sequence_distance.append(value)

            else:
                sequence_index.extend([dictionary[UNK]])
                sequence_distance.append(1.0)

            pre_word = word


        # If the length is longer than the sentence limit, the token is truncated later.
        # print("dictionary pad = ", dictionary[PAD])
        # print("dictionary UNK = ", dictionary[UNK])
        if len(sequence_index) > DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]
            sequence_distance = sequence_distance[:DEFINES.max_sequence_length]

        # You are putting the length in one sentence.
        sequences_length.append(len(sequence_index))
        # sentence length is longer than max_sequence_length
        # If it is small, put PAD (0) in the empty part.
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        sequence_distance += (DEFINES.max_sequence_length -len(sequence_distance))*[dictionary[PAD]]


        # Indexed value
        # Put it in sequences_input_index.
        #print("Sequence_index = ", sequence_index)
        sequences_input_index.append(sequence_index)
        sequence_input_distance.append(sequence_distance)

    # Convert a regular indexed array to a NumPy array.
    # The reason for putting it in the TensorFlow dataset
    # It is pre-work.
    # Arrays indexed to NumPy arrays
    # Give the length.
    # For time sequence construction

    max_distance = max([max(x) for x in sequence_input_distance])
    sequence_input_distance = np.asarray([[y / max_distance for y in x] for x in sequence_input_distance])

    return np.asarray(sequences_input_index), np.asarray(sequence_input_distance), np.array(sequence_input_user), sequences_length


# The value key to index is word and the value
# Receive an index dictionary.
def dec_output_processing(value, dictionary):
    # Holding index values
    # Array (cumulative)
    sequences_output_index = []
    # One decoding input sentence
    # Has a length (accumulates)
    sequences_length = []
    # Whether to use morpheme toning

    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # Blows line by line.

    for sequence in value:
        # FILTERS = "([~.,!? \" ':;) (]) "
        # Using normalization, the filter contains
        # Replace values ​​with "".
        sequence = re.sub(CHANGE_FILTER, "", sequence)

        # Take it when decoding a sentence
        # Array to be.
        sequence_index = []
        # Because the START must come at the beginning of the decoding input
        # Enter the value and start.
        # Get a word by space unit in a sentence,
        # Insert the index that is the value.
        sequence_index = [dictionary[STD]] + [dictionary[word] for word in sequence.split()]

        # If the length is longer than the sentence limit, the token is truncated later.
        if len(sequence_index) > DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length]
            #You are putting the length in one sentence.
        sequences_length.append (len (sequence_index))
        # sentence length is longer than max_sequence_length
        # If it is small, put PAD (0) in the empty part.
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
    # Indexed value
        # Insert sequences_output_index.
        sequences_output_index.append(sequence_index)
    #  Convert a regular indexed array to a NumPy array.
    # The reason for putting it in the TensorFlow dataset
    # It is pre-work.
    # Pass the indexed array and its length to the NumPy array.
    return np.asarray(sequences_output_index), sequences_length


def padding(x, length):
    x_new = np.zeros([len(x), length])
    for i, y in enumerate(x):
        if len(y) <= length:
            x_new [i, 0:len(y)] = y
    return np.asarray(x_new)


def padding_user(x, length):
    x_new = np.zeros([len(x), length])
    for i, y in enumerate(x):
        x_new = x[i]*length
    return np.asarray(x_new)

def make_output_embedding(x):

    x_new = np.zeros([x.shape[0], x.shape[1]])
    for i, y in enumerate(x):
        x_new[i,0] = 1
        x_new[i, 1:len(y)] = y[0:len(y)-1]

    return np.asarray(x_new)


# The value and key to be indexed are words
# Take a dictionary whose value is an index.
def dec_target_processing(value, dictionary):
    # Holding index values
    # Array (cumulative)
    sequences_target_index = []
    # Whether to use morpheme toning
    if DEFINES.tokenize_as_morph:
        value = prepro_like_morphlized(value)
    # Blows line by line.
    for sequence in value:
        # FILTERS = "([~.,!? \" ':;) (]) "
        # Using normalization, the filter contains
        # Replace values ​​with "".
        sequence = re.sub(CHANGE_FILTER, "", sequence)
        # Bring word by space unit in sentence
        # Enter the index, which is the value of the dictionary.
        # Put END at the end of decoding output.
        sequence_index = [dictionary[word] for word in sequence.split()]
        # If the length is longer than the sentence limit, the token is truncated later.
        # And put END token


        if len(sequence_index) >= DEFINES.max_sequence_length:
            sequence_index = sequence_index[:DEFINES.max_sequence_length-1] + [dictionary[END]]
        else:
            sequence_index += [dictionary[END]]


        # sentence length is longer than max_sequence_length
        # If it is small, put PAD (0) in the empty part.
        sequence_index += (DEFINES.max_sequence_length - len(sequence_index)) * [dictionary[PAD]]
        # Indexed value
        # Put it in sequences_target_index.
        sequences_target_index.append(sequence_index)
    # Convert a regular indexed array to a NumPy array.
    # The reason is a preliminary work to put in the TensorFlow dataset.
    # Pass the indexed array and its length to the NumPy array.
    return np.asarray(sequences_target_index)



# Distance between two points logitutes and latitudes
def euclidean_dist (point1, point2):
    (lat1, lon1) = point1
    (lat2, lon2) = point2
    radius = 6371000  # m

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d



def rearrange(input, in_distance, in_time, in_users, in_cate, in_poisim, output, out_dist, out_t, out_users, target):
    features = {"input": input, "in_distance":in_distance, "in_time":in_time, "in_users":in_users, "in_cate":in_cate, "in_poisim":in_poisim, "output": output, "out_t":out_t, "out_distance":out_dist, "out_users": out_users}
    return features, target

# This is a function that goes into learning and creates batch data.
def train_input_fn(train_input, train_out, train_target_dec, batch_size):

    (train_input_enc, train_input_dist, train_input_time, train_input_users, train_input_cate, train_input_poisim) =  train_input
    (train_output_dec, train_output_dist, train_output_time, train_output_users) = train_out

    # As part of creating dataset, from_tensor_slices part
    # Cut each sentence into one sentence.
    # train_input_enc, train_output_dec, train_target_dec
    # Divide the three into one sentence each.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_input_dist, train_input_time, train_input_users, train_input_cate, train_input_poisim, train_output_dec, train_output_dist, train_output_time, train_output_users, train_target_dec))
    # Decay entire data
    dataset = dataset.shuffle(buffer_size=len(train_input_enc))
    # If there is no batch argument value, an error occurs.
    assert batch_size is not None, "train batchSize must not be None"
    # Sharing through from_tensor_slices
    # Bundle by batch size.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # For each element of data, use the rearrange function Convert elements through  # and compose them into maps.
    dataset = dataset.map(rearrange)
    # If you can put the desired number of epochs in the repeat () function,
    # If there are no arguments, iterators are infinite.
    dataset = dataset.repeat()

    # iterator through make_one_shot_iterator
    # Make it.
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) #dataset.make_one_shot_iterator()
    # Tensor of next item via iterator
    # Give the object.

    return iterator.get_next()


## This is a function that goes into evaluation and creates batch data.
def eval_input_fn(eval_input, eval_out, eval_target_dec, batch_size):
    (eval_input_enc, eval_input_dist, eval_input_time, eval_input_users,eval_input_cate,eval_input_poisim) = eval_input
    (eval_output_dec, eval_output_dist, eval_output_time, eval_out_users) = eval_out

    # As part of creating dataset, from_tensor_slices part
    # Cut each sentence into one sentence.
    # eval_input_enc, eval_output_dec, eval_target_dec
    # Divide the three into one sentence each.
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_input_dist, eval_input_time,eval_input_users, eval_input_cate, eval_input_poisim, eval_output_dec, eval_output_dist, eval_output_time, eval_out_users, eval_target_dec))
    # Shuffle entire data
    dataset = dataset.shuffle(buffer_size=len(eval_input_enc))
    # If there is no batch argument value, an error occurs.
    assert batch_size is not None, "eval batchSize must not be None"
    # Sharing through from_tensor_slices
    # Bundle by batch size.
    dataset = dataset.batch(batch_size, drop_remainder=True)
    # For each element of data, use the rearrange function Convert elements through  # and compose them into maps.
    dataset = dataset.map(rearrange)
    # If you can put the desired number of epochs in the repeat () function,
    # If there are no arguments, iterators are infinite.
    # As it is an evaluation, it is operated only once.
    dataset = dataset.repeat(1)

    # via make_one_shot_iterator
    # Create an iterator.
    iterator = tf.compat.v1.data.make_one_shot_iterator(dataset) # dataset.make_one_shot_iterator()
    # Through the iterator,
    # Give the tensor object.

    return iterator.get_next()

def findPopularityFromData(dataset):

    global POPULARITY

    dfvisits1 = pd.read_excel('DataExcelFormat/userVisits-' + dataset + '-allPOI.xlsx')

    dfvisits1.drop_duplicates(subset=['nsid', 'poiID', 'seqID'])

    POPULARITY = to_frequency_table(dfvisits1.poiID)


def  makeDoc2VecData(dataset):
    if dataset in "Gowall,Foursquare":
        descriptoin = pd.read_excel('DataExcelFormatWithPOIDescription/POI-" + dataset +"_"+str(POI_number)+ ".xlsx')
    else:
        descriptoin = pd.read_excel('DataExcelFormatWithPOIDescription/POI-' + dataset + 'withDescription.xlsx')
    questionForm = pd.DataFrame(columns=["id","qid1","qid2","question1","question2","is_duplicate"])
    index = 0
    for i in range(len(descriptoin)):
        for j in range(len(descriptoin)):
            if descriptoin.iloc[i].theme == descriptoin.iloc[j].theme:
                questionForm.at[questionForm.shape[0]] = [index, descriptoin.iloc[i].poiID, descriptoin.iloc[j].poiID,descriptoin.iloc[i].description, descriptoin.iloc[j].description,1]
            else:
                questionForm.at[questionForm.shape[0]] = [index, descriptoin.iloc[i].poiID, descriptoin.iloc[j].poiID,
                                                           descriptoin.iloc[i].description,
                                                           descriptoin.iloc[j].description, 0]
        index = index + 1
    return questionForm

def category_paragraph_sequence(input, C, S, idx2char):
    new_C = np.zeros([input.shape[0],input.shape[1]])
    new_S = np.zeros([input.shape[0],input.shape[1]])

    for i in range(input.shape[0]):
        new_C[i,0] = 1
        new_S[i,0] = 1
        for j in range(1, input.shape[1]):
            if input[i][j] != 0:
                new_S[i, j] = S[idx2char[input[i, j - 1]], idx2char[input[i, j]]]
                new_C[i, j] = C[idx2char[input[i, j - 1]], idx2char[input[i, j]]]
            else:
                break

    return new_C, new_S


def evaluation_distance(value,targets, idx2char, cordinates,POI_des_sim):


    index = 0
    popularity_5 = 0
    popularity_10 = 0
    distance_5, distance_10 = 0, 0
    interest_5, interest_10 = 0.0, 0.0

    for v in value:
        pop_5, pop_10 = 0,0
        index2 = 0

        target_poi = targets[index][0]
        # tarPOP = POPULARITY[idx2char[target_poi]]

        index_topk, score_topk = largest_indices(v['logit'][index2], 10)
        top_k = np.asarray(index_topk)
        dis_5, dis_10 = 0.0, 0.
        inter_5, inter_10 = 0.0, 0.0
        for x in top_k[0:5]:
            dis_5 += euclidean_dist(cordinates[idx2char[x]], cordinates[idx2char[target_poi]])
            inter_5 += POI_des_sim[(idx2char[target_poi],idx2char[x])]


        for x in top_k:
            dis_10 += euclidean_dist(cordinates[idx2char[x]], cordinates[idx2char[target_poi]])
            inter_10 += POI_des_sim[(idx2char[target_poi], idx2char[x])]

        dis_5 = dis_5 / 5
        dis_10 = dis_10 / 10

        inter_5 = inter_5 / 5
        inter_10 = inter_10 / 10

        popularity_5 += pop_5
        popularity_10 += pop_10
        distance_5 += dis_5
        distance_10 += dis_10

        interest_5 += inter_5
        interest_10 += inter_10


        index = index + 1

    distance_5 /= index
    distance_10 /= index

    interest_5 /= index
    interest_10 /= index


    return  distance_5, distance_10, interest_5, interest_10




def main(dataset):

    start_time = timeit.default_timer()
    pre_5, pre_10, f1_5, f1_10, recall_5, recall_10, ndcg_5, ndcg_10  = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    results = pd.DataFrame(columns=['pre_5', 'pre_10', 'f1_5', 'f1_10', 'recall_5', 'recall_10', 'ndcg_5', 'ndcg_10'])
    iteration_number = 10
    train_execution_time, test_execution_time = 0, 0
    data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)

    os.makedirs(data_out_path, exist_ok=True)

    # POPULARITY =
    findPopularityFromData(dataset)




    max_len, time = preprocess(dataset)
    Cordinates, Category = findCordinates_Category(dataset)
    # print("Category =", Category)

    data = makeDoc2VecData(dataset)

    # Take POI similarity from the categories based similarity
    POI_des_sim = d2v.Doc2Vec_Similarity(data)
    # print("POI similiarty = ", POI_des_sim)
    DEFINES.max_sequence_length = min(max_len, DEFINES.max_sequence_length)


    char2idx, idx2char, vocabulary_length = load_vocabulary(dataset)

    DEFINES.vocabulary_size = vocabulary_length

    for i in range(1, DEFINES.vocabulary_size+1):
        if 'POI_'+str(i) not in POPULARITY:
            POPULARITY['POI_'+str(i)] = 1

    # Import training data and test data.

    train_input, train_label, eval_input, eval_label, test_input, test_label, train_time, eval_time, user, timeid = load_data(dataset,time)

    (train_user, eval_user) = user

    # This is the part of creating a training set encoding.
    train_input_enc, train_input_dist, train_input_users, train_input_enc_length = enc_processing(train_input,train_user, char2idx,Cordinates)

    train_input_time = padding(train_time, DEFINES.max_sequence_length)
    train_output_dec, train_output_dec_length = dec_output_processing(train_label, char2idx)
    train_output_dist = make_output_embedding(train_input_dist)
    train_output_time = make_output_embedding(train_input_time)
    train_output_users = make_output_embedding(train_input_users)

    train_target_dec = dec_target_processing(train_label, char2idx)

    # This is the part of making evaluation set encoding.
    eval_input_enc, eval_input_dist, eval_input_users, eval_input_enc_length = enc_processing(eval_input, eval_user,char2idx, Cordinates)
    eval_input_time = padding(eval_time, DEFINES.max_sequence_length)

    eval_output_dec, eval_output_dec_length = dec_output_processing(eval_label, char2idx)
    eval_output_dist = make_output_embedding(eval_input_dist)
    eval_output_time = make_output_embedding(eval_input_time)

    eval_output_users = make_output_embedding(eval_input_users)
    eval_target_dec = dec_target_processing(eval_label, char2idx)

    user_length = max(max([max(y) for y in eval_input_users]), max([max(y) for y in train_input_users])) + 1

    # print("train input enc = ", train_input_enc)
    cate_input_enc, train_POI_similarity = category_paragraph_sequence(train_input_enc, Category, POI_des_sim, idx2char)
    cate_eval_enc, eval_POI_similarity = category_paragraph_sequence(eval_input_enc, Category, POI_des_sim, idx2char)

    # Find the start time

    for i in range(iteration_number):
        #Clear the existing models
        check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
        os.makedirs(check_point_path, exist_ok=True)

        if (os.path.exists(DEFINES.check_point_path)):
            clearExistingFile()

        # Make up an estimator.
        classifier = tf.estimator.Estimator(
            model_fn=Model,  # Register the model.
            model_dir=DEFINES.check_point_path,  # Register checkpoint location.
            params={  # Pass parameters to the model.
                'hidden_size': DEFINES.hidden_size,  # Set the weight size.
                'learning_rate': DEFINES.learning_rate,  # Set learning rate.
                'vocabulary_length': vocabulary_length,  # Sets the dictionary size.
                'embedding_size': DEFINES.embedding_size,  # Set the embedding size.
                'max_sequence_length': DEFINES.max_sequence_length,
                'user_length': user_length,

            })


        # # Learning run

        train_input = (train_input_enc, train_input_dist, train_input_time, train_input_users, cate_input_enc, train_POI_similarity)
        train_output = (train_output_dec, train_output_dist, train_output_time, train_output_users)

        eval_input = (eval_input_enc, eval_input_dist, eval_input_time, eval_input_users,cate_eval_enc,eval_POI_similarity)
        eval_output = (eval_output_dec, eval_output_dist, eval_output_time, eval_output_users)

        start_training_time = timeit.default_timer()
        classifier.train(input_fn=lambda: train_input_fn(train_input, train_output, train_target_dec, DEFINES.batch_size),steps=DEFINES.train_steps)
        train_execution_time = train_execution_time + timeit.default_timer() - start_training_time

        start_test_time = timeit.default_timer()
        eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(eval_input, eval_output, eval_target_dec, DEFINES.batch_size), steps=1)
        print("Iterations = ", i + 1)
        print('\nEVAL set precision_5:{precision_5:0.3f} recall_5:{recall_5:0.3f} precision_10:{precision_10:0.3f} recall_10:{recall_10:0.3f}  ndcg_5:{ndcg_5:0.3f}  ndcg_10:{ndcg_10:0.3f}'.format(
                **eval_result))
        test_execution_time = test_execution_time + timeit.default_timer() - start_test_time
        c_pre_5, c_recall_5, c_f1_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_5, c_ndcg_10 = '{precision_5:0.3f},{recall_5:0.3f}, {f1_5:0.3f}, {precision_10:0.3f},{recall_10:0.3f},{f1_10:0.3f}, {ndcg_5:0.3f},{ndcg_10:0.3f}'.format(
            **eval_result).split(",")



        pre_5 = pre_5 + float(c_pre_5)
        pre_10 = pre_10 + float(c_pre_10)
        f1_5 = f1_5 + float(c_f1_5)
        f1_10 = f1_10 + float(c_f1_10)
        recall_5 = recall_5 + float(c_recall_5)
        recall_10 = recall_10 + float(c_recall_10)
        ndcg_5 = ndcg_5 + float(c_ndcg_5)
        ndcg_10 = ndcg_10 + float(c_ndcg_10)




        results.at[results.shape[0]] = [c_pre_5, c_pre_10, c_f1_5, c_f1_10, c_recall_5, c_recall_10, c_ndcg_5, c_ndcg_10]

    pre_5 = pre_5 / iteration_number
    pre_10 = pre_10 / iteration_number
    f1_5 = f1_5 / iteration_number
    f1_10 = f1_10 / iteration_number
    recall_5 = recall_5 / iteration_number
    recall_10 = recall_10 / iteration_number
    ndcg_5 = ndcg_5 / iteration_number
    ndcg_10 = ndcg_10 / iteration_number


    print("The test data total_precison_5 is %f, total_precison_10 is %f" % (pre_5, pre_10))
    print("The test data total_recall_5 is %f, total_recall_10 is %f" % (recall_5, recall_10))
    print("The test data total_f1_5 is %f, total_f1_10 is %f" % (f1_5, f1_10))
    print("The test data total_ncdd_5 is %f, total_ndcg_10 is %f" % (ndcg_5, ndcg_10))


    if (DEFINES.user_interest == 1):
        results.to_excel('Results_Final_DMKD/TLR-UI_results_POIDes' + dataset + '.xlsx')
    else:
        results.to_excel('Results_Final_DMKD/TLR-UI_results_Cate' + dataset + '.xlsx')


    end_time =timeit.default_timer()

    print("Time Duration = ", end_time - start_time, "seconds")
    execute_time = end_time - start_time
    return execute_time, train_execution_time, test_execution_time

if __name__ == '__main__':

    executive_time = pd.DataFrame(columns=['Dataset', 'total_time', 'train_time', 'test_time'])
    datasets = ["epcot","MagicK","caliAdv","Buda","Edin","Melbourne"]

    for data in datasets:
        # Run main function based  dataset

        ex_time, train_time, test_time = main(data)
        print("Time Duration = ", ex_time, " seconds", "Training time = ", train_time, " test time = ", test_time)
        executive_time.at[executive_time.shape[0]] = [data, ex_time, train_time, test_time]
    if (DEFINES.user_interest == 1):
        executive_time.to_excel('Results_Final_DMKD/TLR-UI_POIDes_Executive_time.xlsx')
    else:
        executive_time.to_excel('Results_Final_DMKD/TLR-UI_Cate_Executive_time.xlsx')

