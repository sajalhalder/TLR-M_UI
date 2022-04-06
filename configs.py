#-*- coding: utf-8 -*-
import tensorflow.compat.v1 as tf

tf.app.flags.DEFINE_integer ('batch_size', 32, 'batch size') # batch size
tf.app.flags.DEFINE_integer ('train_steps', 200, 'train steps') # learning epoch
tf.app.flags.DEFINE_float ('dropout_width', 0.5, 'dropout width') # Dropout size
tf.app.flags.DEFINE_integer ('hidden_size', 128, 'weights size') # weight size # use paper 512
tf.app.flags.DEFINE_integer ('vocabulary_size', 0, 'vocab size') # weight size # use paper 512
tf.app.flags.DEFINE_float ('learning_rate', 1e-3, 'learning rate') # Learning rate
tf.app.flags.DEFINE_integer ('shuffle_seek', 100, 'shuffle random seek') # shuffle seed value
tf.app.flags.DEFINE_integer ('max_sequence_length', 25, 'max sequence length') # Sequence length
tf.app.flags.DEFINE_integer ('embedding_size', 128, 'embedding size') # embedding size # paper 512 using learning speed and performance tuning
tf.app.flags.DEFINE_integer ('query_dimention', 128, 'q # uery dimention') # Paper 512 Use Learning Speed ​​and Performance Tuning
tf.app.flags.DEFINE_integer ('key_dimention', 128, 'key dimention') # Use paper 512 Learning speed and performance tuning
tf.app.flags.DEFINE_integer ('value_dimention', 128, 'value dimention') # Paper 512 Use Learning Speed ​​and Performance Tuning
tf.app.flags.DEFINE_integer ('layers_size', 2, 'layers size') # Use 6 layers or 2 papers to tune learning speed and performance
tf.app.flags.DEFINE_integer ('heads_size', 4, 'heads size') # Papers use 8 headers or 4 learning speed and performance tuning
tf.app.flags.DEFINE_string ('data_path', './data_in/input.csv', 'data path') # Data location
#tf.app.flags.DEFINE_string ('data_path', './data_in/ChatBotData_1.csv', 'data path') # Data location
tf.app.flags.DEFINE_string ('vocabulary_path', './data_out/vocabularyData.voc', 'vocabulary path') # Dictionary location
tf.app.flags.DEFINE_string ('check_point_path', './data_out/check_point', 'check point path') # Checkpoint location
tf.app.flags.DEFINE_boolean ('tokenize_as_morph', False, 'set morph tokenize') # Whether or not tokenizing is used depending on the morpheme
tf.app.flags.DEFINE_boolean ('conv_1d_layer', False, 'set conv 1d layer') # Whether to use the second conv1d among the two methods of the paper
tf.app.flags.DEFINE_boolean ('xavier_embedding', True, 'set init xavier embedding') # Enable or disable embedding using Xavier initialization
tf.app.flags.DEFINE_boolean ('mask_loss', False, 'set masking loss') # Using masking for loss (PAD, END)
tf.app.flags.DEFINE_integer ('max_queue_time', 0, 'max_queue_time') # Using masking for loss (PAD, END)
tf.app.flags.DEFINE_string('test_part','random','last or random')
tf.app.flags.DEFINE_integer('POI_numbers', 1000,'POI_numbers')
tf.app.flags.DEFINE_integer('min_poi', 10,'min_poi')
tf.app.flags.DEFINE_integer('min_user', 10,'min_user')


# For fairRec models
tf.compat.v1.flags.DEFINE_integer('topk', 5, 'topk')  # Dropout size
tf.compat.v1.flags.DEFINE_float('alpha', 0.5, 'alpha')  #
tf.compat.v1.flags.DEFINE_float('min_under_demand',0.0, 'min_under_demand')  # minimum under demand threshold
tf.compat.v1.flags.DEFINE_float('envy_cap', 0.5, 'envy capacity')  # minimum envy free capacity

tf.compat.v1.flags.DEFINE_integer('user_interest', 2, 'POI des or Category' ) # 1 = POI des and 2 = POI Category
# Define FLAGS
DEFINES = tf.app.flags.FLAGS
