from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from scipy.ndimage.interpolation import rotate
import h5py

# Loading the MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Retrieving the datasets and reshaping them
Y_train = mnist.train.images.reshape(mnist.train.images.shape[0],28,28)
Y_valid = mnist.validation.images.reshape(mnist.validation.images.shape[0],28,28)
Y_test = mnist.test.images.reshape(mnist.test.images.shape[0],28,28)

# Rotate the images by some random angle and save them as the training datasets
def rotating(Y):
    X = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        X[i,:,:] = rotate(Y[i,:,:], np.random.randint(1,360), reshape=False)
        if i%1000 == 0:
            print("Number of reshaped frames: ",i)
    return X

print("Generating X_train")
X_train_1 = rotating(Y_train)
X_train_2 = rotating(Y_train)
X_train = np.concatenate([X_train_1, X_train_2], axis = 0)
Y_train = np.concatenate([Y_train, Y_train], axis=0)
print("Generating X_valid")
X_valid_1 = rotating(Y_valid)
X_valid_2 = rotating(Y_valid)
X_valid = np.concatenate([X_valid_1, X_valid_2], axis = 0)
Y_valid = np.concatenate([Y_valid, Y_valid], axis = 0)
print("Generating X_test")
X_test = rotating(Y_test)

# Saving the datasets
with h5py.File("data.hdf5", 'a') as f:
    f.create_dataset("X_train", data=X_train)
    f.create_dataset("Y_train", data=Y_train)
    f.create_dataset("X_valid", data=X_valid)
    f.create_dataset("Y_valid", data=Y_valid)
    f.create_dataset("X_test", data=X_test)
    f.create_dataset("Y_test", data=Y_test)
    f.create_dataset("train_labels", data= mnist.train.labels)
    f.create_dataset("valid_labels", data=mnist.validation.labels)
    f.create_dataset("test_labels", data=mnist.test.labels)
