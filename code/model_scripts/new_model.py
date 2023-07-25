import os 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ["CUDA_VISIBLE_DEVICES"]= "6"
import numpy as np
from netCDF4 import Dataset
import matplotlib.pyplot as plt 
import tensorflow.compat.v1 as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import Conv3D, Conv2D
from keras.layers import ConvLSTM2D
#from keras.layers.normalization import BatchNormalization
from keras.layers import Dropout, Lambda
from keras import optimizers
from keras.layers import Conv2DTranspose,TimeDistributed
from keras import callbacks
import xarray as xr
import keras

#print("##########################give the path where you want to store the model#####################")
lead_time = 0
model_path = "/home/cccr/roxy/matin/MTech_project/model/model/conv-lstm/7in1out/7in_1out.h5"
save_path = "/home/cccr/roxy/matin/MTech_project/model/model/conv-lstm/7in1out/"

def preprocess_data(sequence, n_steps,channels):
       
    #separating nan and non nan points and doing exp space transform
    sequence = sequence.sel(time=slice("1997-01-01","2014-12-31"))
    
    #selecting data for jun,july,aug,sept
    sequence = sequence.where(sequence.time.dt.month.isin([6,7,8,9]), drop=True)
    max = sequence.max()
    min = sequence.min()
    
    #max min normalisation 
    sequence = (sequence-min)/(max - min)
    print(sequence.time.count())
    
    sequence=np.array(sequence)
    
    
    #print(sequence)
    validpts = np.where(sequence==sequence)
    #T = np.max(sequence[validpts])
    #print(channels)
    
    #mean=np.mean(sequence[validpts])
    #std=np.std(sequence[validpts])
    #sequence[validpts] = sequence[validpts]/np.max(sequence[validpts]) # bringing all valid points down to 0-1
    #converting to new normalized space (i.e. exponential space) 
    
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix + lead_time > len(sequence)-1:
            
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix + lead_time]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

def create_input(channels, n_steps):
    stack = []
    for i in range(len(channels)):
        #print("#########################put the path for rainfall and sst data########################################################")
        input, _ = preprocess_data(xr.open_dataarray(f"/home/cccr/roxy/matin/{channels[i]}.nc"), n_steps=steps,channels=channels[i])
        input = np.expand_dims(input,axis=2)
        print(f"Adding channel {i} with shape: {input.shape}")
        stack.append(input)
    out = np.dstack(stack)
    return out
channels=["GPCP_JJAS.nc","sst40yearmean.nc"]
steps=7
input = create_input(channels, steps)
#print("########################put the path of rainfall data#########################################################")
_, target = preprocess_data(xr.open_dataarray("/home/cccr/roxy/matin/GPCP_JJAS.nc"), n_steps=steps,channels=channels[-1])
target = np.expand_dims(target, axis=1)

#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@


x=np.rollaxis(input,4,2)
x=np.rollaxis(x,4,2)
print(x.shape)

y=np.rollaxis(target,3,1)
y=np.rollaxis(y,3,1)
print(y.shape)


assert input.shape[0] == target.shape[0]
assert input.shape[-1] == target.shape[-1]
assert input.shape[-2] == target.shape[-2]

n_rows, n_cols = input.shape[-2], input.shape[-1]

print("INPUT SHAPE --> ", input.shape)
print("TARGET SHAPE --> ", target.shape)

del input
del target

seq = tf.keras.Sequential()

seq.add(ConvLSTM2D(filters=4,kernel_size=(3,3),padding='same',input_shape=(2,137,201,2),return_sequences=True,data_format='channels_last'))
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
Adam = tf.keras.optimizers.Adam(lr=10**-4)
seq.compile(loss='mean_squared_error',optimizer=Adam,metrics=['mae'])

print(seq.summary())
logdir = save_path

#Train the model...
nepoch=300
tbcallback = callbacks.TensorBoard(log_dir=logdir, histogram_freq=10, batch_size=10, write_graph=True)
reduce_lr =tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=20, min_lr=10**-20)

check_point = callbacks.ModelCheckpoint(
    model_path,
    # save_weights_only=True,
    monitor='val_mae',
    mode='min',
    save_best_only=True,
)
check_point2 = callbacks.ModelCheckpoint(
    model_path,
    # save_weights_only=True,
    monitor='val_loss',
    mode='min',
    save_best_only=True,
)
# In[14]:

early_stop = tf.keras.callbacks.EarlyStopping(patience=30, restore_best_weights=True)

history=seq.fit(x,y,epochs=nepoch,validation_split=0.1,batch_size=30,callbacks=[early_stop,check_point,check_point2,tbcallback,reduce_lr,tf.keras.callbacks.TerminateOnNaN()])
seq.save(model_path)
np.save(f'{save_path}7in_1out.npy',history.history)

# In[19]:

