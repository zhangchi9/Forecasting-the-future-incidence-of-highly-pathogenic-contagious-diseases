import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Lambda, SpatialDropout1D
from tensorflow.keras.layers import Conv1D, LSTM, Multiply, Add, Dense, Dropout
from python.tcn import TCN


def model_LSTM(n_feat):
    model = Sequential()
    model.add(LSTM(units=16, return_sequences=True, input_shape=(n_steps, n_feat)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(units=16))
    model.add(Dropout(0.5))
    model.add(Dense(units=1))

    return model


    
def model_TCN(n_his, n_feat, n_out=1, dropout_rate = 0.2):
    # convolutional operation parameters
    n_filters = 32 # 32 
    filter_width = 2
    dilation_rates = [2**i for i in range(4)] * 2 # 8
    dropout_rate = dropout_rate

    # define an input history series and pass it through a stack of dilated causal convolution blocks. 
    history_seq = Input(shape=(n_his, n_feat))
    x = history_seq

    skips = []
    for dilation_rate in dilation_rates:
        
        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(32, 1, padding='same', activation='relu')(x)
        
        # filter convolution
        x_f = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='causal',
                     dilation_rate=dilation_rate)(x)
        
        # gating convolution
        x_g = Conv1D(filters=n_filters,
                     kernel_size=filter_width, 
                     padding='causal',
                     dilation_rate=dilation_rate)(x)
        
        # multiply filter and gating branches
        z = Multiply()([Activation('tanh')(x_f),
                        Activation('sigmoid')(x_g)])
        
        # postprocessing - equivalent to time-distributed dense
        z = Conv1D(32, 1, padding='same', activation='relu')(z)
        z = Dropout(dropout_rate)(z)
        
        # residual connection
        x = Add()([x, z])    
        
        # collect skip connections
        skips.append(z)

    # add all skip connection outputs 
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers 
    out = Conv1D(128, 1, padding='same')(out)
    out = Activation('relu')(out)
    out = Dropout(dropout_rate)(out)
    out = Conv1D(1, 1, padding='same')(out)

    # extract the last 60 time steps as the training target
    def slice(x, seq_length):
        return x[:,-seq_length:,:]

    pred_seq_train = Lambda(slice, arguments={'seq_length':n_out})(out)

    model = Model(history_seq, pred_seq_train)
    return model



def model_Keras_TCN(n_his, n_feat):
    input_layer = Input(shape=(n_his, n_feat))

    x = TCN(nb_filters=16, 
            kernel_size=2, 
            nb_stacks=2, 
            dilations=(1, 2, 4, 8), 
            padding='causal',
            use_skip_connections=True, 
            dropout_rate=0.2, 
            return_sequences=False,
            activation='relu', 
            kernel_initializer='he_normal', 
            use_batch_norm=False, 
            use_layer_norm=False,
            name='tcn')(input_layer)

    output_layer = Dense(1)(x)
    
    model = Model(input_layer, output_layer)
    return model



