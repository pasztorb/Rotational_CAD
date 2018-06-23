from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Conv2DTranspose, Flatten,Reshape, UpSampling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
import h5py
import numpy as np

def getModel_1():
    input_img = Input(shape=(1,28,28))
    x = Conv2D(4,(3,3),
               activation='relu',
               data_format='channels_first')(input_img)
    x = Conv2D(8,(3,3),
               activation='relu',
               data_format='channels_first')(x)
    x = Flatten()(x)
    code = Dense(512)(x)
    x = Dense(4608)(code)
    x = Reshape((8,24,24))(x)
    x = Conv2DTranspose(4, (3,3),
                        activation='relu',
                        data_format='channels_first')(x)
    decoded = Conv2DTranspose(1, (3,3),
                        activation='sigmoid',
                        data_format='channels_first')(x)

    model = Model(input_img, decoded)
    return model

def getModel_2():
    input_img = Input(shape=(1, 28, 28))
    x = Flatten()(input_img)
    x = Dense(512)(x)
    x = Dense(256)(x)
    x = Dense(512)(x)
    x = Dense(784)(x)
    output = Reshape((1,28,28))(x)
    model = Model(input_img, output)

    return model

def getModel_3():
    input_img = Input(shape=(1,28,28))
    x = Conv2D(8,(3,3),
               activation='relu',
               padding='same',
               data_format='channels_first')(input_img)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(16,(3,3),
               activation='relu',
               padding='same',
               data_format='channels_first')(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    code = Dense(512)(x)
    x = Dense(784)(code)
    x = Reshape((16,7,7))(x)
    x = UpSampling2D((2,2),
                     data_format='channels_first')(x)
    x = Conv2D(4, (3,3),
               activation='relu',
               padding='same',
               data_format='channels_first')(x)
    x = UpSampling2D((2,2),
                     data_format='channels_first')(x)
    decoded = Conv2D(1, (3,3),
                     padding='same',
                     activation='sigmoid',
                     data_format='channels_first')(x)

    model = Model(input_img, decoded)
    return model


def train(model):
    # Load datasets
    with h5py.File('data.hdf5','r') as f:
        X_train = f['X_train'][()]
        Y_train = f['Y_train'][()]
        X_valid = f['X_valid'][()]
        Y_valid = f['Y_valid'][()]

    # Add the extra channel axis
    X_train = X_train[:,np.newaxis,:,:]
    Y_train = Y_train[:, np.newaxis, :, :]
    X_valid = X_valid[:, np.newaxis, :, :]
    Y_valid = Y_valid[:, np.newaxis, :, :]

    # Callbacks
    modelcp = ModelCheckpoint("saved-model-{epoch:02d}-{val_loss:.4f}.hdf5",
                              save_best_only=True,
                              mode='min')
    earlystop = EarlyStopping(monitor='val_loss', patience=5)


    model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid),
              epochs=50,
              batch_size=128,
              shuffle=True,
              callbacks=[modelcp, earlystop],
              verbose=2)

    model.save('model.hdf5')

model = getModel_3()
print(model.summary())
model.compile(optimizer='adam', loss='mse')
train(model)