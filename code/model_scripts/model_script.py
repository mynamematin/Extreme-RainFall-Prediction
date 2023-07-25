
import os 
#os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
#os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, ConvLSTM2D
from tensorflow.keras import optimizers
from tensorflow.keras import callbacks
#channels = CONFIG["FEATURES_NAMES"]
#num_channels = len(channels)
#ds_path = CONFIG["COMBINED_PATH"]
#step = int(CONFIG["N_STEPS"])

def preprocess_data(sequence, n_steps,channels):
    #selecting data for jun,july,aug,sept
    sequence = sequence.where(sequence.time.dt.month.isin([6,7,8,9]), drop=True)    
    #separating nan and non nan points and doing exp space transform
    sequence=np.array(sequence)
    
    
    #print(sequence)
    validpts = np.where(sequence==sequence)
    #T = np.max(sequence[validpts])
    #print(channels)
    
    mean=np.mean(sequence[validpts])
    std=np.std(sequence[validpts])
    sequence[validpts] = (sequence[validpts]-mean)/std # bringing all valid points down to 0-1
    #converting to new normalized space (i.e. exponential space) 
    sequence[validpts] = np.exp(sequence[validpts])
    sequence[validpts] = np.power(sequence[validpts],7)
    
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_input(channels, n_steps):
    stack = []
    for i in range(len(channels)-1):
        input, _ = preprocess_data(xr.open_dataarray('/lus/dal/erpas_rnd/Abhijit/data/{}.nc'.format(channels[i])), n_steps=steps,channels=channels[i])
        input = np.expand_dims(input, axis=2)
        print(f"Adding channel {i} with shape: {input.shape}")
        stack.append(input)
    out = np.dstack(stack)
    return out
channels=["u10_15_20","v10_15_20","mx2t_15_20","mn2t_15_20","t2m_15_20","q_15_20","tp_15_20"]
steps=5
input = create_input(channels, steps)
_, target = preprocess_data(xr.open_dataarray("/lus/dal/erpas_rnd/Abhijit/data/t2m_15_20.nc"), n_steps=steps,channels=channels[-1])
target = np.expand_dims(target, axis=1)



x=np.rollaxis(input,4,2)
x=np.rollaxis(x,4,2)
x.shape

y=np.rollaxis(target,3,1)
y=np.rollaxis(y,3,1)
y.shape


assert input.shape[0] == target.shape[0]
assert input.shape[-1] == target.shape[-1]
assert input.shape[-2] == target.shape[-2]

n_rows, n_cols = input.shape[-2], input.shape[-1]

print("INPUT SHAPE --> ", input.shape)
print("TARGET SHAPE --> ", target.shape)


seq = keras.Sequential()

seq.add(ConvLSTM2D(filters=4,kernel_size=(3,3),padding='same',input_shape=(5, n_rows, n_cols,len(channels)-1),return_sequences=True,data_format='channels_last'))
#seq.add(TimeDistributed(Conv2DTranspose(filters=20,kernel_size=(3,3),padding='valid',activation='relu')))
seq.add(ConvLSTM2D(filters=8,kernel_size=(3,3),padding='same',return_sequences=True,data_format='channels_last'))
#seq.add(TimeDistributed(Conv2DTranspose(filters=40,kernel_size=(3,3),padding='valid',activation='relu')))
seq.add(ConvLSTM2D(filters=8,kernel_size=(3,3),padding='same',return_sequences=True,data_format='channels_last'))
#seq.add(TimeDistributed(Conv2DTranspose(filters=60,kernel_size=(3,3),padding='valid',activation='relu')))
seq.add(ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=True,data_format='channels_last'))
seq.add(ConvLSTM2D(filters=16,kernel_size=(3,3),padding='same',return_sequences=False,data_format='channels_last'))
seq.add(Conv2D(filters=15,kernel_size=(3,3),activation='relu',padding='same',data_format='channels_last'))
seq.add(Conv2D(filters=1,kernel_size=(3,3),activation='relu',padding='same',data_format='channels_last'))
#seq.add(Conv3DTranspose(filters=1,kernel_size=(1,1,1),activation='relu',padding='valid'))

Adam = optimizers.Adam(learning_rate=10**-4)
seq.compile(loss='mean_squared_error',optimizer=Adam,metrics=['mae'])

print(seq.summary())
logdir = "/lus/dal/erpas_rnd/Abhijit/results/"

#Train the model...
nepoch=2
tbcallback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=10, batch_size=8, write_graph=True)
reduce_lr =tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=30, min_lr=0.001)

check_point = callbacks.ModelCheckpoint(
    "/lus/dal/erpas_rnd/Abhijit/results/",
    # save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True,
)
check_point2 = callbacks.ModelCheckpoint(
    "/lus/dal/erpas_rnd/Abhijit/results/",
    # save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
)
# In[14]:

history=seq.fit(x[:380],y[:380],epochs=nepoch,validation_split=0.1,batch_size=8,callbacks=[check_point,check_point2,tbcallback,reduce_lr,tf.keras.callbacks.TerminateOnNaN()])
seq.save("/lus/dal/erpas_rnd/Abhijit/results/")
np.save("/lus/dal/erpas_rnd/Abhijit/results/",history.history)

# In[19]:




