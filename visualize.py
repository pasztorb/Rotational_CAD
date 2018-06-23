import sys
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import numpy as np

model_path = sys.argv[1]
model = load_model(model_path)

output_path = sys.argv[2]

num_im = 5

f = h5py.File('data.hdf5','r')

# Plot training images
train_index = np.random.randint(0,f['X_train'].shape[0]-1, num_im)
for i in train_index:
    # Plot the input data
    X = f['X_train'][i,:,:]
    plt.imshow(X)
    plt.savefig(output_path+'/X_train_'+str(i))
    # Plot the target data
    plt.imshow(f['Y_train'][i,:,:])
    plt.savefig(output_path+'/Y_train_'+str(i))
    # Plot the predicted image
    plt.imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:])
    plt.savefig(output_path+'/train_pred_'+str(i))

# Plot validation images
valid_index = np.random.randint(0,f['X_valid'].shape[0]-1, num_im)
for i in valid_index:
    # Plot the input data
    X = f['X_valid'][i,:,:]
    plt.imshow(X)
    plt.savefig(output_path+'/X_valid_'+str(i))
    # Plot the target data
    plt.imshow(f['Y_valid'][i,:,:])
    plt.savefig(output_path+'/Y_valid_'+str(i))
    # Plot the predicted image
    plt.imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:])
    plt.savefig(output_path+'/valid_pred_'+str(i))

# Plot validation imagess
test_index = np.random.randint(0,f['X_test'].shape[0]-1, num_im)
for i in valid_index:
    # Plot the input data
    X = f['X_test'][i,:,:]
    plt.imshow(X)
    plt.savefig(output_path+'/X_test_'+str(i))
    # Plot the target data
    plt.imshow(f['Y_test'][i,:,:])
    plt.savefig(output_path+'/Y_test_'+str(i))
    # Plot the predicted image
    plt.imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:])
    plt.savefig(output_path+'/test_pred_'+str(i))


f.close()