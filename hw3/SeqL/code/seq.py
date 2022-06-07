import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
import tensorflow as tf
Max_S = 10
BATCH_S = 4
col = 8 # 8,9
class Local(tf.keras.layers.Layer):
    def __init__(self, latent_dim=64, intermediate_dim=128, name='Local', **kwargs):
        super(Local, self).__init__(name=name, **kwargs)
        self.cnn = tf.keras.layers.Conv2D(filters=64, kernel_size=(1,col), padding='valid', 
                                                 input_shape=(7,col,1), activation='relu', name = 'localcnn')
        self.bn = tf.keras.layers.BatchNormalization(name='localbn')
        self.dropout = tf.keras.layers.Dropout(0.2, name='localdrop')
        self.reshape = tf.keras.layers.Reshape((7,64), name = 'localreshape')
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False, stateful=False, name = 'locallstm')
        self.dense_1 = tf.keras.layers.Dense(intermediate_dim, activation='relu', name = 'localdense1')
        self.dense_2 = tf.keras.layers.Dense(latent_dim, activation='relu',name = 'localdense2')
        
    def call(self, inputs):
        cnn_bn = self.bn(self.cnn(inputs))
        cnn_drop = self.dropout(cnn_bn)
        lstm_inp = self.reshape(cnn_drop)
        #print (lstm_inp.shape)
        lstm_out = self.lstm(lstm_inp)
        lstm_drop = self.dropout(lstm_out)
        dense_1 = self.dropout(self.dense_1(lstm_drop))
        local_out = self.dense_2(dense_1)
        return local_out
    
    def get_config(self):
        config = super(Local, self).get_config()
        return config
    

class Attention(tf.keras.layers.Layer):
    def __init__(self, attention_size = 32, name='Attention',**kwargs):
            super().__init__(name=name, **kwargs)
            self.attention_size = attention_size
            self.dense_1 = tf.keras.layers.Dense(attention_size, activation = 'tanh', name = 'attndnese1') 
            self.dense_2 = tf.keras.layers.Dense(1, use_bias = False, name = 'attndense2')

    def call(self, inputs):
        v = self.dense_1(inputs)
        vu = self.dense_2(v)
        alphas = tf.nn.softmax(vu)
        output = alphas
        return output
    
    def get_config(self):
        config = super(Attention, self).get_config()
        config.update({"attention_size":self.attention_size})
        return config

import AttentionCell
from imp import reload
reload(AttentionCell)

class Global(tf.keras.layers.Layer):
    def __init__(self, max_subs = Max_S, latent_dim = 64, name='Global',**kwargs):
        super(Global, self).__init__(name = name,**kwargs)
        self.max_subs = max_subs

        self.cell = AttentionCell.ALSTM_Cell(64, name= 'attncell')
        self.rnn = tf.keras.layers.RNN(self.cell, return_sequences = True,name = 'attnRNN')
        self.dense_1 = tf.keras.layers.Dense(64, activation = 'relu',name='globaldense1') #tf.keras.layers.BatchNormalization(),
        self.dense_2 = tf.keras.layers.Dense(32, activation = 'relu',name =  'globaldense2')
        self.out = tf.keras.layers.Dense(2, activation = 'sigmoid', name = 'globalout')
        self.attn = Attention(name='globalattn')
        
    def call(self, inputs):
        x = inputs
        #mask = self.mask(x)
        h = self.rnn(x)
        h_d_1 = self.dense_1(h)
        h_d_2 = self.dense_2(h_d_1)
        h_a = self.attn(h_d_2)
        y = self.out(h_d_2 * h_a)
        return y
    
    def get_config(self):
        config = super(Global, self).get_config()
        config.update({"max_subs": self.max_subs})
        return config

mode_class_num=3

class Seq(tf.keras.Model):
    def __init__(self, latent_dim = 64, intermediate_dim = 128, name='Seq', **kwargs):
        super().__init__(name=name, **kwargs)
        self.local_out_dim = latent_dim
        self.local = Local(latent_dim=latent_dim, intermediate_dim = intermediate_dim, name='seqlocal')
        self.pred = Global(name='seqglobal')
        
    def call(self, inputs):
        batch_inputs = inputs[0]
        batch_time = inputs[1]

        x = self.local(batch_inputs)
        x = tf.reshape(x, [ BATCH_S, Max_S, self.local_out_dim])
        batch_time = tf.reshape(batch_time, [BATCH_S, Max_S, 1])
        x = tf.keras.layers.concatenate([x, batch_time], axis=-1)
        x = self.pred(x)
        
        return x
    
    def get_config(self):
        config = super(Seq, self).get_config()
        config.update({"local_out_dim": self.local_out_dim})
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)


