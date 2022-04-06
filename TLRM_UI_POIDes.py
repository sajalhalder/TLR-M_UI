"""
Research Article: POI Recommendation with Queuing Time and User Interest Awareness (In Submission DMKD)
Authors: Sajal Halder, Jeffrey Chan, Xiuzhen Zhang and Kwan Hui Lim
Implemented By: Sajal Halder, PhD Candidate, RMIT University, Australia
Implementation Time: November 2022 - May 2021
Description: POI description based users interest in top-k POI recommendation and queuing time prediction
"""
import tensorflow as tf
# import tensorflow_ranking as tfr
from konlpy.tag import Twitter
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from datetime import datetime
import pandas as pd
import pathlib
import re
import random
import glob
from tqdm import tqdm
import shutil
import math
import random, collections, itertools
from scipy import stats
from itertools import groupby
import warnings
import Doc2Vec as d2v
import timeit
warnings.filterwarnings("ignore")
tf.compat.v1.disable_v2_behavior()
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
QTIME = {}
Cordinates = {}
POPULARITY = {}

from configs import DEFINES
DATA_OUT_PATH = './data_out/'
loss_alpha = 0.5

# dataset = 'Melbourne'#, Edin, Buda, caliAdv, MagicK, epcot


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


#
def encoder(inputs,  num_units, heads, num_layers):
    outputs = inputs

    for _ in range(num_layers):
        outputs = encoder_module(outputs, num_units, heads)

    return outputs


def encoder2(inputs, inputs2, num_units, heads, num_layers):
    outputs = inputs
    outputs2 = inputs2
    for _ in range(num_layers):
        outputs = encoder_module(outputs, num_units, heads)
        outputs2 = encoder_module(outputs2, num_units, heads)
        outputs = outputs + outputs2

    return outputs, outputs2


def decoder(inputs, encoder_outputs, num_units, heads, num_layers):
    outputs = inputs
    for _ in range(num_layers):
        outputs = decoder_module(outputs, encoder_outputs, num_units, heads)

    return outputs


def decoder2(inputs, inputs2, encoder_outputs, encoder_outputs_queue, num_units, heads, num_layers):
    outputs = inputs
    outputs2 = inputs2
    for _ in range(num_layers):
        outputs = decoder_module(outputs, encoder_outputs, num_units, heads)
        outputs2 = decoder_module(outputs2, encoder_outputs_queue, num_units, heads)
        outputs = outputs + outputs2

    return outputs, outputs2



def find_idcg():
    idcg_5, idcg_10 = 0.0, 0.0

    for i in range(5):
        idcg_5 = idcg_5 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    for i in range(10):
        idcg_10 = idcg_10 + tf.math.log(2.0) / tf.math.log(float(i) + 2)

    return idcg_5, idcg_10


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



    embedding = tf.compat.v1.get_variable(name='embedding', shape=[params['vocabulary_length'], params['embedding_size']],dtype=tf.float32, initializer=tf.initializers.GlorotUniform()) #tf.layers.xavier_initializer())
    W = tf.compat.v1.tile(tf.expand_dims(tf.compat.v1.get_variable('W', [1, params['embedding_size']], dtype=tf.float32), 0),[DEFINES.batch_size, 1, 1])
    embedding_user1 = tf.compat.v1.get_variable("embedding_user", shape = [params['user_length'], params['embedding_size']], dtype=tf.float32, initializer= tf.initializers.GlorotUniform()) #tf.contrib.layers.xavier_initializer())
    W_u = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_u", [params['embedding_size'], params['embedding_size']], dtype=tf.float32),0), [DEFINES.batch_size, 1, 1])

    W_p = tf.tile(tf.expand_dims(tf.compat.v1.get_variable("W_p", [params['embedding_size'], params['embedding_size']], dtype= tf.float32), 0), [DEFINES.batch_size, 1, 1])


    encoder_inputs = tf.matmul(tf.nn.embedding_lookup(embedding, features['input']),W_p)
    decoder_inputs = tf.matmul(tf.nn.embedding_lookup(embedding, features['output']),W_p)
    encoder_features = tf.matmul(tf.expand_dims(tf.cast(features['in_distance'], tf.float32), 2), W) + tf.matmul(tf.expand_dims(tf.cast(features['in_time'], tf.float32), 2), W)

    encoder_features_UI =  tf.matmul(tf.expand_dims(tf.cast(features['in_poisim'], tf.float32), 2), W) #* 0.5 + tf.matmul(tf.expand_dims(tf.cast(features['in_cate'], tf.float32), 2), W) * 0.5


    encoder_user_features = tf.matmul(tf.nn.embedding_lookup(embedding_user1, features['in_users']),W_u)
    position_encode = tf.nn.embedding_lookup(positional_encoded, position_inputs)

    encoder_inputs  += encoder_features + encoder_user_features + encoder_features_UI
    encoder_inputs = encoder_inputs + position_encode
    decoder_inputs = decoder_inputs + position_encode

    encoder_queue_features = tf.matmul(tf.expand_dims(tf.cast(features['in_queue'], tf.float32), 2), W)
    decoder_queue_features = tf.matmul(tf.expand_dims(tf.cast(features['out_queue'], tf.float32), 2), W)

    encoder_inputs_queue = encoder_inputs + encoder_queue_features + position_encode
    decoder_inputs_queue = decoder_inputs + position_encode

    encoder_outputs= encoder(encoder_inputs, [params['hidden_size'] * 4, params['hidden_size']], DEFINES.heads_size, DEFINES.layers_size)
    decoder_outputs = decoder(decoder_inputs, encoder_outputs, [params['hidden_size'] * 4, params['hidden_size']],DEFINES.heads_size, DEFINES.layers_size)

    encoder_outputs_queue = encoder(encoder_inputs_queue, [params['hidden_size'] * 4, params['hidden_size']], DEFINES.heads_size,
                              DEFINES.layers_size)
    decoder_outputs_queue = decoder(decoder_inputs_queue, encoder_outputs_queue, [params['hidden_size'] * 4, params['hidden_size']],
                              DEFINES.heads_size, DEFINES.layers_size)

    # encoder_outputs, encoder_outputs_queue = encoder2(encoder_inputs, encoder_inputs_queue,
    #                                                   [params['hidden_size'] * 4, params['hidden_size']],
    #                                                   DEFINES.heads_size, DEFINES.layers_size)
    # decoder_outputs, decoder_outputs_queue = decoder2(decoder_inputs, decoder_inputs_queue, encoder_outputs,
    #                                                   encoder_outputs_queue,
    #                                                   [params['hidden_size'] * 4, params['hidden_size']],
    #                                                   DEFINES.heads_size, DEFINES.layers_size)

    logits = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs + decoder_outputs_queue)

    predict = tf.argmax(logits[:, :, :], -1)
    logits_queue = tf.keras.layers.Dense(params['vocabulary_length'])(decoder_outputs_queue)

    # Update logits based on queue information
    logits = logits + logits_queue
    predict_queue = tf.nn.top_k(logits_queue[:, :, :], 1)[0]  # tf.argmax(logits_queue[:,:,:], -1)

    if PREDICT:
        predictions = {
            'topk': tf.nn.top_k(logits[:, 0:1, :], 10)[1],  # tf.nn.top_k(logits[:,0:1,:],10)[1]
            'logit': logits,
            'reward': tf.nn.top_k(logits[:, 0:1, :], 10)[0]

        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)
   

    label1 = labels['target']
    label2 = labels['t_queue']

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label1))
    loss_queue = tf.reduce_mean(tf.keras.losses.mean_squared_error(predict_queue, tf.cast(label2,tf.float32)))


    loss2 = loss_alpha * loss + (1 - loss_alpha) * loss_queue  # tf.group(loss, loss_queue) # K.mean(loss + loss_queue)
    #loss2 = loss + loss_queue
    if EVAL:
        correct_prediction_5 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.tile(label1[:,0:1],[1,5]), tf.int32), tf.nn.top_k(logits[:,0:1,:],5)[1]), tf.float32))*5 #/ tf.cast((tf.shape(logits)[0] * tf.shape(logits)[1]),tf.float32) #DEFINES.batch_size * params['vocabulary_length'])
        correct_prediction_10 = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.tile(label1[:, 0:1], [1, 10]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 10)[1]),tf.float32)) *10
        recall_5 = (correct_prediction_5, correct_prediction_5)
        precision_5 = (correct_prediction_5/5, correct_prediction_5/5)
        recall_10 = (correct_prediction_10, correct_prediction_10)
        precision_10 = (correct_prediction_10/10, correct_prediction_10/10)
        f1_5 = (2*recall_5[0]*precision_5[0] /(recall_5[0] + precision_5[0] + 1e-8), 2*recall_5[0]*precision_5[0] /(recall_5[0] + precision_5[0] + 1e-8))
        f1_10 = (2 * recall_10[0]* precision_10[0] / (recall_10[0] + precision_10[0] + 1e-8), 2 * recall_10[0]* precision_10[0] / (recall_10[0] + precision_10[0] + 1e-8))


        idcg_5, idcg_10 = find_idcg()
        ndcg_5 = tf.reduce_mean(tf.math.log(2.0) / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(tf.cast(tf.tile(label1[:, 0:1], [1, 1]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 5)[1]), tf.int64)),tf.float32) + 2.0)  ))  / idcg_5 #* tf.cast(DEFINES.batch_size, tf.float32))#
        ndcg_10 = tf.reduce_mean(tf.math.log(2.0) / (tf.math.log(tf.cast(tf.where(tf.cast(tf.equal(tf.cast(tf.tile(label1[:, 0:1], [1, 1]), tf.int32), tf.nn.top_k(logits[:, 0:1, :], 10)[1]), tf.int64)),tf.float32) + 2.0) ) )/ idcg_10 #* tf.cast(DEFINES.batch_size, tf.float32))

        ndcg_5 = (ndcg_5, ndcg_5)
        ndcg_10 = (ndcg_10, ndcg_10)

        rmse = tf.sqrt(tf.reduce_mean(tf.math.squared_difference(tf.cast(predict_queue * params['max_queue'], tf.float32), tf.cast(label2 * params['max_queue'], tf.float32))))
        rmse = (rmse, rmse)


        metrics = {'recall_5': recall_5, 'precision_5': precision_5, 'f1_5':f1_5, 'recall_10': recall_10, 'precision_10': precision_10, 'f1_10':f1_10,'ndcg_5': ndcg_5, 'ndcg_10': ndcg_10, 'rmse':rmse}

        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    assert TRAIN

    # lrate = d−0.5 *  min(step_num−0.5, step_num · warmup_steps−1.5)
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    train_op1 = optimizer.minimize(loss_queue, global_step=tf.compat.v1.train.get_global_step())

    train_op2 = tf.group(train_op, train_op1)

    return tf.estimator.EstimatorSpec(mode, loss=loss2, train_op=train_op2)

    # # lrate = d−0.5 *  model · min(step_num−0.5, step_num · warmup_steps−1.5)
    # optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=DEFINES.learning_rate)
    # train_op = optimizer.minimize(loss, global_step=tf.compat.v1.train.get_global_step())
    #
    # return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

# sequence based time distance
def time_function(dataset, dfVisits, train_pois):
    # print("dfVisits = ", dfVisits)
    # print("Train pois = ", train_pois)

    train_time = []
    train_queue = []
    train_time.extend([1])  # initially 100 second for first POI
    train_queue.extend([1])  # initially 100 second for first POI
    dfVisit1 = dfVisits.sort_values('takenUnix', ascending=True).drop_duplicates('poiID').reset_index()
    dfVisit2 = dfVisits.sort_values('takenUnix', ascending=False).drop_duplicates('poiID').reset_index()
    dfVisit2 = dfVisit2.sort_values('takenUnix', ascending=True)
    # print("dfVisits1 = ", dfVisit1)
    # print("dfVisits2 = ", dfVisit2)
    for i in range(1, len(train_pois)):

        t = dfVisit1[dfVisit1.poiID == train_pois[i]].takenUnix.values[0] - dfVisit1[dfVisit1.poiID == train_pois[i-1]].takenUnix.values[0] + 1
        hour = datetime.fromtimestamp(dfVisit1[dfVisit1.poiID == train_pois[i]].takenUnix.values[0]).hour
        if dataset in ['epcot', 'disland','MagicK','caliAdv','disHolly']:
            q = QTIME[(int(train_pois[i]),int(hour))]#
        else:
            q  = dfVisit1[dfVisit1.poiID == train_pois[i]].takenUnix.values[0] - dfVisit2[dfVisit2.poiID == train_pois[i-1]].takenUnix.values[0]  + 1
        #print("t = ", t , " q = ",q)
        train_queue += [q/60]
        train_time += [t/60]

    return train_time, train_queue




def preprocess(dataset):
    global QTIME
    min_poi = 3
    min_user = 3

    dfVisits = pd.read_excel('DataExcelFormat/userVisits-' + dataset + '-allPOI.xlsx')
    if dataset in ["epcot", "disland", "MagicK", "caliAdv", 'disHolly']:
        dfQueue = pd.read_excel('DataExcelFormat/queueTimes-'+dataset+'.xlsx')
        for i in range(dfQueue.shape[0]):
            poi_q, hour, t_q, = int(dfQueue.iloc[i]['poiID']), int(dfQueue.iloc[i]['hour']),dfQueue.iloc[i]['avgQueueTime']
            QTIME[(poi_q,hour)] = t_q
        # print("Q time = ", QTIME)
    dfVisits = dfVisits[dfVisits.takenUnix > 0]
    dfVisits['user_freq'] = dfVisits.groupby('nsid')['nsid'].transform('count')
    dfVisits = dfVisits[dfVisits.user_freq >= min_user]
    dfVisits['poi_freq'] = dfVisits.groupby('poiID')['poiID'].transform('count')
    dfVisits = dfVisits[dfVisits.poi_freq >= min_poi]

    dfVisits = dfVisits[['nsid', 'poiID', 'takenUnix','seqID']]
    # print(dfVisits)
    df = pd.DataFrame(columns=['Train', 'Test', 'U','T','Q','Q_t'])



    train_part = 0.7
    sequences = dfVisits.seqID.unique()
    # print(sequences)
    max_len = 0
    for seq in sequences:
        tempdfVisits = dfVisits[dfVisits.seqID == seq]
        tempdfVisits = tempdfVisits.sort_values(['takenUnix'], ascending=[True])
        user = tempdfVisits.iloc[0].nsid
        pois = [i[0] for i in groupby(tempdfVisits.poiID)] #.unique()
        #if len(pois) >= min_poi:
        if user not in USERID:
            USERID[user] = len(USERID)
        if len(tempdfVisits) == 0:
            continue

        userid = USERID[user]

        max_len = len(pois)-1 if max_len < len(pois)-1 else max_len
        time, queue = time_function(dataset,tempdfVisits, pois)
        for i in range(len(pois) - 1):

            startW = 0 #max(i - 2 * window_size, 0)
            endW = i + 2  #
            train_d = ['POI_' + str(poi) for poi in pois[startW:endW-1]]
            test_d = ['POI_' + str(poi) for poi in pois[endW-1:endW]]

            train_time = [t for t in time[startW:endW-1]]
            train_queue = [t for t in queue[startW:endW - 1]]
            target_time = [t for t in time[endW-1:endW]]
            target_queue = [t for t in queue[endW - 1:endW]]
            userseq = [int(userid) for i in range(len(train_d))]
            train_time[0] = 1

            df.at[df.shape[0]] = (train_d, test_d, userseq, train_time, train_queue, target_queue)
            # train_time.append(time)


    characters = ["'", ",", "[", "]"]
    for char in characters:
        df['Train'] = df['Train'].apply(str).str.replace(char, '')
        df['Test'] = df['Test'].apply(str).str.replace(char, '')
        df['U'] = df['U'].apply(str).str.replace(char, '')
        df['T'] = df['T'].apply(str).str.replace(char, '')
        df['Q'] = df['Q'].apply(str).str.replace(char, '')
        df['Q_t'] = df['Q_t'].apply(str).str.replace(char, '')


    #Save input sequence
    df.to_csv('data_in/input_'+dataset+'_train_TLRM.csv', index=False)
    # Travel time sequence normalization
    # Recent check has most important

    # max_time = max([max(x) for x in train_time])  # max([max(x) for x in time])
    # train_time = [[y / max_time for y in x] for x in train_time]

    return int(max_len) #, train_time



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
            if dfNodes.iloc[i].theme == dfNodes.iloc[j].theme:
                Category['POI_'+str(dfNodes.iloc[i].poiID), 'POI_'+str(dfNodes.iloc[j].poiID)] = 1
            else:
                Category['POI_'+str(dfNodes.iloc[i].poiID), 'POI_'+str(dfNodes.iloc[j].poiID)] = 0

    return Cordinats, Category
#**********************************************************************************************************************


def padding(x, length):
    x_new = np.zeros([len(x), length])
    for i, y in enumerate(x):
        y = y.split(" ")
        if len(y) <= length:
            x_new [i, 0:len(y)] = y
    return np.asarray(x_new)


def padding1(x, length):

    x_new = np.zeros([len(x), length])
    for i, y in enumerate(x):
        y = y.split(" ")
        if len(y) <= length:
            x_new [i, 0:len(y)] = y
    return np.asarray(x_new)


def make_output_embedding(x):

    x_new = np.zeros([x.shape[0], x.shape[1]])
    for i, y in enumerate(x):
        x_new[i,0] = 1
        x_new[i, 1:len(y)] = y[0:len(y)-1]

    return np.asarray(x_new)


def load_data(dataset):

    path = 'data_in/input_'+dataset+'_train_TLRM.csv'
    data_df = pd.read_csv(path, header=0)

    question, answer, user, time, queue, queue_target = list(data_df['Train']), list(data_df['Test']), list(data_df['U']), list(data_df['T']), list(data_df['Q']), list(data_df['Q_t'])
    random_seed = random.randint(10,100)


    #print(random_seed)
    train_input, eval_input1, train_label, eval_label1 = train_test_split(question, answer, test_size=0.30 , random_state=random_seed)
    train_user, eval_user = train_test_split(user,  test_size=0.30, random_state= random_seed)
    # devide time based on test size
    train_time, eval_time, train_queue, eval_queue = train_test_split(time, queue, test_size = 0.30, random_state= random_seed)
    eval_input, test_input, eval_label, test_label = train_test_split(eval_input1, eval_label1, test_size=0.66, random_state=random_seed+1)
    train_target_queue, eval_target_queue = train_test_split(queue_target, test_size= 0.30, random_state= random_seed)


    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    #Sort train_input based on sequence length
    sorted_index = len_argsort(train_input)
    train_input = [train_input[i] for i in sorted_index]
    train_label = [train_label[i] for i in sorted_index]
    train_time = [train_time[i] for i in sorted_index]
    train_user = [train_user[i] for i in sorted_index]

    train_queue = [train_queue[i] for i in sorted_index]
    train_target_queue = [train_target_queue[i] for i in sorted_index]

    # Sort eval_input based on sequence length
    sorted_index = len_argsort(eval_input1)
    eval_input1 = [eval_input1[i] for i in sorted_index]
    eval_label1 = [eval_label1[i] for i in sorted_index]
    eval_time = [eval_time[i] for i in sorted_index]
    eval_user = [eval_user[i] for i in sorted_index]

    eval_queue = [eval_queue[i] for i in sorted_index]
    eval_target_queue = [eval_target_queue[i] for i in sorted_index]

    user = (train_user, eval_user)

    queue = (train_queue, eval_queue)
    target_queue = ( train_target_queue, eval_target_queue)

    return train_input, train_label, eval_input1, eval_label1, test_input, test_label, train_time, eval_time, user,  queue, target_queue

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


def load_vocabulary(dataset):
    # Prepare an array to hold the dictionary
    vocabulary_list = []
    # After configuring the dictionary, proceed to save it as a file.
    # Check the existence of the file.
    path = 'data_in/input_'+dataset+'_train_TLRM.csv'
    if (os.path.exists(path)):
        # Through the judgment because data exists
        # Load data
        data_df = pd.read_csv(path, encoding='utf-8')

        # Through the data frame of Pandas
        # Bring columns for questions and answers.
        question, answer = list(data_df['Train']), list(data_df['Test'])
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
        # All about common words
        # No need to make one
        # Set and list them.
        words = list(set(words))

        # MARKER in advance without data
        # To add, process as below.
        # Below is the MARKER value, from the first in the list
        # Add to index 0 to put in order.
        # PAD = "<PADDING>"
        # STD = "<START>"
        # END = "<END>"
        # UNK = "<UNKNWON>"
        words[:0] = MARKER
        #print("Words 1= ", words)
        # Since we have made a list of dictionaries,
        # Create a dictionary file.


        with open(DEFINES.vocabulary_path, 'w', encoding='utf-8') as vocabulary_file:
            for word in words:

                vocabulary_file.write(word + '\n')

    # If dictionary file exists here
    # Load the file and put it in an array.
    with open(DEFINES.vocabulary_path, 'r', encoding='utf-8') as vocabulary_file:
        for line in vocabulary_file:
            vocabulary_list.append(line.strip())

    # The contents of the array have keys and values
    # Make it a dictionary structure.
    char2idx, idx2char = make_vocabulary(vocabulary_list)
    # Returns two types of keys and values.
    # (Example) word: index, index: word)
    return char2idx, idx2char, len(char2idx)


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


# The value and key to be indexed are words
# Take a dictionary whose value is an index.
def enc_processing(value, users, dictionary, cordinates):
    # Holding index values
    #print("Value = ", value)
    # Array (cumulative)
    sequences_input_index = []
    sequence_input_distance = []
    sequence_input_time = []
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
        sequence_time = []

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

def rearrange(input, in_distance, in_time, in_queue, in_users, in_cate, in_poisim, output, out_dist, out_t, out_queue, out_users, target, target_queue):
    features = {"input": input, "in_distance":in_distance, "in_time":in_time, "in_queue":in_queue, "in_users":in_users, "in_cate":in_cate, "in_poisim":in_poisim, "output": output, "out_t":out_t, "out_distance":out_dist, "out_queue":out_queue, "out_users": out_users}
    labels = {"target": target, "t_queue": target_queue}
    return features, labels


# This is a function that goes into learning and creates batch data.
def train_input_fn(train_input, train_out, train_target_dec, train_target_queue, batch_size):

    (train_input_enc, train_input_dist, train_input_time, train_input_users, train_input_queue,train_input_cate, train_input_poisim) =  train_input
    (train_output_dec, train_output_dist, train_output_time, train_output_users, train_output_queue) = train_out

    # As part of creating dataset, from_tensor_slices part
    # Cut each sentence into one sentence.
    # train_input_enc, train_output_dec, train_target_dec
    # Divide the three into one sentence each.
    dataset = tf.data.Dataset.from_tensor_slices((train_input_enc, train_input_dist, train_input_time, train_input_queue, train_input_users, train_input_cate, train_input_poisim, train_output_dec, train_output_dist, train_output_time, train_output_queue, train_output_users, train_target_dec, train_target_queue))
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
def eval_input_fn(eval_input, eval_out, eval_target_dec, eval_target_queue, batch_size):
    (eval_input_enc, eval_input_dist, eval_input_time, eval_input_users, eval_input_queue, eval_input_cate, eval_input_poisim) = eval_input
    (eval_output_dec, eval_output_dist, eval_output_time, eval_out_users, eval_output_queue) = eval_out

    # As part of creating dataset, from_tensor_slices part
    # Cut each sentence into one sentence.
    # eval_input_enc, eval_output_dec, eval_target_dec
    # Divide the three into one sentence each.
    dataset = tf.data.Dataset.from_tensor_slices((eval_input_enc, eval_input_dist, eval_input_time, eval_input_queue, eval_input_users, eval_input_cate, eval_input_poisim, eval_output_dec, eval_output_dist, eval_output_time, eval_output_queue, eval_out_users, eval_target_dec, eval_target_queue))
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
    iterator =  tf.compat.v1.data.make_one_shot_iterator(dataset)  # dataset.make_one_shot_iterator()
    # Through the iterator,
    # Give the tensor object.

    return iterator.get_next()


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

def to_frequency_table(data):
    frequencytable = {}
    for key in data:
        if 'POI_'+ str(key) in frequencytable:
            frequencytable['POI_'+str(key)] += 1
        else:
            frequencytable['POI_'+str(key)] = 1
    return frequencytable


def findPopularityFromData(dataset):

    global POPULARITY

    dfvisits1 = pd.read_excel('DataExcelFormat/userVisits-' + dataset + '-allPOI.xlsx')

    dfvisits1.drop_duplicates(subset=['nsid', 'poiID', 'seqID'])

    POPULARITY = to_frequency_table(dfvisits1.poiID)

def  makeDoc2VecData(dataset):
    descriptoin = pd.read_excel('DataExcelFormatWithPOIDescription/POI-' + dataset + 'withDescription.xlsx')
    questionForm = pd.DataFrame(columns=["id","qid1","qid2","question1","question2","is_duplicate"])
    index = 0
    for i in range(len(descriptoin)):
        for j in range(i, len(descriptoin)):
            if descriptoin.iloc[i].theme == descriptoin.iloc[j].theme:
                questionForm.at[questionForm.shape[0]] = [index, descriptoin.iloc[i].poiID, descriptoin.iloc[j].poiID,descriptoin.iloc[i].description, descriptoin.iloc[j].description,1]
            else:
                questionForm.at[questionForm.shape[0]] = [index, descriptoin.iloc[i].poiID, descriptoin.iloc[j].poiID,descriptoin.iloc[i].description, descriptoin.iloc[j].description, 0]
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
                # print(input[i, j - 1], input[i, j])
                # print(S[idx2char[input[i, j - 1]], idx2char[input[i, j]]])
                new_S[i, j] = S[idx2char[input[i, j - 1]], idx2char[input[i, j]]] if (idx2char[input[i, j - 1]], idx2char[input[i, j]]) in S else S[idx2char[input[i, j]], idx2char[input[i, j-1]]]
                new_C[i, j] = C[idx2char[input[i, j - 1]], idx2char[input[i, j]]]
            else:
                break

    return new_C, new_S


def main(dataset):

    pre_5, pre_10, f1_5, f1_10, recall_5, recall_10, ndcg_5, ndcg_10, pop5, pop10, dis5, dis10, total_rmse = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    results = pd.DataFrame(columns=['pre_5', 'pre_10', 'f1_5', 'f1_10', 'recall_5', 'recall_10', 'ndcg_5', 'ndcg_10','rmse'])

    iteration_number = 10

    train_execution_time, test_execution_time = 0,0

    data_out_path = os.path.join(os.getcwd(), DATA_OUT_PATH)
    os.makedirs(data_out_path, exist_ok=True)

    findPopularityFromData(dataset)



    Cordinates, Category = findCordinates_Category(dataset)
    # print("Category =", Category)

    data = makeDoc2VecData(dataset)

    # ************* For POI description based similarity  *************************
    if (DEFINES.user_interest == 1):
        POI_des_sim = d2v.Doc2Vec_Similarity(data)

    # ********************** End POI Description based simialrity
    else:
        POI_des_sim = Category

    # print("POI Similarity Done ")

    max_len = preprocess(dataset)



    DEFINES.max_sequence_length = max_len+1

    #dep.themepark_or_city()

    char2idx, idx2char, vocabulary_length = load_vocabulary(dataset)

    DEFINES.vocabulary_size = vocabulary_length

    for i in range(1, DEFINES.vocabulary_size + 1):
        if 'POI_' + str(i) not in POPULARITY:
            POPULARITY['POI_' + str(i)] = 1

    train_input, train_label, eval_input, eval_label, test_input, test_label, train_time, eval_time, user, queue , target_queue = load_data(dataset)
    (train_user, eval_user)  = user
    (train_queue, eval_queue) = queue
    (train_target_queue, eval_target_queue) = target_queue

    # This is the part of creating a training set encoding.
    train_input_enc, train_input_dist,train_input_users, train_input_enc_length = enc_processing(train_input, train_user, char2idx, Cordinates)

    # Make the sequence fixed length using 0 padding
    train_input_time = padding(train_time, DEFINES.max_sequence_length)
    train_input_queue = padding(train_queue, DEFINES.max_sequence_length)

    # print("train time = ", train_input_time)
    # print("train input queue = ", train_input_queue)

    # Make one bit shifted decoder inputs
    train_output_dec, train_output_dec_length = dec_output_processing(train_label, char2idx)
    train_output_dist = make_output_embedding(train_input_dist)
    train_output_time = make_output_embedding(train_input_time)
    train_output_queue = make_output_embedding(train_input_queue)
    train_output_users = make_output_embedding(train_input_users)
    train_target_dec = dec_target_processing(train_label, char2idx)

    # This is the part of making evaluation set encoding.
    eval_input_enc, eval_input_dist, eval_input_users, eval_input_enc_length = enc_processing(eval_input,eval_user, char2idx,Cordinates)
    eval_input_time = padding(eval_time,DEFINES.max_sequence_length)
    eval_input_queue = padding(eval_queue, DEFINES.max_sequence_length)

    # Decoder evaluation inputs
    eval_output_dec, eval_output_dec_length = dec_output_processing(eval_label, char2idx)
    eval_output_dist = make_output_embedding(eval_input_dist)
    eval_output_time = make_output_embedding(eval_input_time)
    eval_output_queue = make_output_embedding(eval_input_queue)

    eval_output_users = make_output_embedding(eval_input_users)
    eval_target_dec = dec_target_processing(eval_label, char2idx)


    user_length = max(max([max(y) for y in eval_input_users]), max([max(y) for y in train_input_users]))+1
    # Find max queue time and normalize based on 0-1
    max_queue = max(max([max(y) for y in train_input_queue]), max([max(y) for y in eval_input_queue]))
    # print(max_queue)

    train_input_queue = np.asarray([[y / max_queue for y in x] for x in train_input_queue])
    eval_input_queue = np.asarray([[y/ max_queue for y in x] for x in eval_input_queue])
    train_output_queue = np.asarray([[y / max_queue for y in x] for x in train_output_queue])
    eval_output_queue = np.asarray([[y / max_queue for y in x] for x in eval_output_queue])
    # print(train_target_queue)
    # print(eval_target_queue)
    train_target_queue = np.asarray([y / max_queue for y in train_target_queue])
    eval_target_queue = np.asarray([y / max_queue for y in eval_target_queue])

    DEFINES.max_queue_time = max_queue

    cate_train_enc, train_POI_similarity = category_paragraph_sequence(train_input_enc, Category, POI_des_sim, idx2char)
    cate_eval_enc, eval_POI_similarity = category_paragraph_sequence(eval_input_enc, Category, POI_des_sim, idx2char)
    start_time = timeit.default_timer()
    for i in range(iteration_number):
    # Import training data and test data.
        # Clear the existing models
        check_point_path = os.path.join(os.getcwd(), DEFINES.check_point_path)
        os.makedirs(check_point_path, exist_ok=True)

        # If model already build clean the directory
        if (os.path.exists(DEFINES.check_point_path)):
            clearExistingFile()

        # Make up an estimator.
        classifier = tf.estimator.Estimator(
            model_fn= Model,  # Register the model.
            model_dir=DEFINES.check_point_path,  # Register checkpoint location.
            params={  # Pass parameters to the model.
                'hidden_size': DEFINES.hidden_size,  # Set the weight size.
                'learning_rate': DEFINES.learning_rate,  # Set learning rate.
                'vocabulary_length': vocabulary_length, # Sets the dictionary size.
                'embedding_size': DEFINES.embedding_size,  # Set the embedding size.
                'max_sequence_length': DEFINES.max_sequence_length,
                'user_length': user_length,
                'max_queue': max_queue,

            })

        # print("classifier = ", classifier)
        # # Learning run

        train_input = (train_input_enc, train_input_dist, train_input_time, train_input_users, train_input_queue, cate_train_enc, train_POI_similarity)
        train_output = (train_output_dec, train_output_dist, train_output_time, train_output_users, train_output_queue)

        eval_input = (eval_input_enc, eval_input_dist, eval_input_time, eval_input_users, eval_input_queue, cate_eval_enc, eval_POI_similarity)
        eval_output = (eval_output_dec, eval_output_dist, eval_output_time, eval_output_users, eval_output_queue)

        start_training_time = timeit.default_timer()
        classifier.train(input_fn=lambda: train_input_fn(train_input, train_output, train_target_dec, train_target_queue, DEFINES.batch_size), steps=DEFINES.train_steps)
        train_execution_time = train_execution_time + timeit.default_timer()-start_training_time

        start_test_time = timeit.default_timer()
        eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(eval_input, eval_output, eval_target_dec, eval_target_queue, DEFINES.batch_size),steps=1)
        test_execution_time = test_execution_time + timeit.default_timer() - start_test_time
        print('\nEVAL set precision_5:{precision_5:0.3f} recall_5:{recall_5:0.3f} precision_10:{precision_10:0.3f} recall_10:{recall_10:0.3f}  ndcg_5:{ndcg_5:0.3f}\n ndcg_10:{ndcg_10:0.3f}'.format(**eval_result))

        print("Iterations = ", i + 1)
        c_pre_5, c_recall_5, c_f1_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_5, c_ndcg_10, rmse = '{precision_5:0.6f},{recall_5:0.6f}, {f1_5:0.6f}, {precision_10:0.6f},{recall_10:0.6f},{f1_10:0.6f}, {ndcg_5:0.6f},{ndcg_10:0.6f}, {rmse:0.6f}'.format(**eval_result).split(",")
        print(c_pre_5, c_recall_5, c_f1_5, c_pre_10, c_recall_10, c_f1_10, c_ndcg_5, c_ndcg_10, rmse)



        pre_5 = pre_5 + float(c_pre_5)
        pre_10 = pre_10 + float(c_pre_10)
        f1_5 = f1_5 + float(c_f1_5)
        f1_10 = f1_10 + float(c_f1_10)
        recall_5 = recall_5 + float(c_recall_5)
        recall_10 = recall_10 + float(c_recall_10)
        ndcg_5 = ndcg_5 + float(c_ndcg_5)
        ndcg_10 = ndcg_10 + float(c_ndcg_10)


        total_rmse = total_rmse + float(rmse)

        results.at[results.shape[0]] = [c_pre_5, c_pre_10, c_f1_5, c_f1_10, c_recall_5, c_recall_10, c_ndcg_5, c_ndcg_10,rmse]

    pre_5 = pre_5 / iteration_number
    pre_10 = pre_10 / iteration_number
    f1_5 = f1_5 / iteration_number
    f1_10 = f1_10 / iteration_number
    recall_5 = recall_5 / iteration_number
    recall_10 = recall_10 / iteration_number
    ndcg_5 = ndcg_5 / iteration_number
    ndcg_10 = ndcg_10 / iteration_number


    total_rmse = total_rmse / iteration_number



    print("The test data total_precison_5 is %f, total_precison_10 is %f" % (pre_5, pre_10))
    print("The test data total_recall_5 is %f, total_recall_10 is %f" % (recall_5, recall_10))
    print("The test data total_f1_5 is %f, total_f1_10 is %f" % (f1_5, f1_10))
    print("The test data total_ncdd_5 is %f, total_ndcg_10 is %f" % (ndcg_5, ndcg_10))

    print("total_rmse = ", total_rmse)

    if (DEFINES.user_interest == 1):
        results.to_excel('Results_Final/TLRM-UI_results_POIDes' + dataset+'.xlsx')
    else:
        results.to_excel('Results_Final/TLRM-UI_results_Cate' + dataset + '.xlsx')

    end_time = timeit.default_timer()
    return end_time-start_time, train_execution_time, test_execution_time

if __name__ == '__main__':

    executive_time = pd.DataFrame(columns=['Dataset','total_time','train_time','test_time'])
    datasets = ["epcot","MagicK","caliAdv","Buda","Edin","Melbourne"]

    for data in datasets:


        # Run main function based  dataset

        ex_time, train_time, test_time =  main(data)
        print("Time Duration = ", ex_time, " seconds", "Training time = ", train_time, " test time = ", test_time)
        executive_time.at[executive_time.shape[0]] = [data,ex_time,train_time, test_time]
    if (DEFINES.user_interest == 1):
        executive_time.to_excel('Results_Final/TLRM-UI_POIDes_Executive_time.xlsx')
    else:
        executive_time.to_excel('Results_Final/TLRM-UI_Cate_Executive_time.xlsx')
