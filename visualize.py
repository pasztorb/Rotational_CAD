import sys
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import numpy as np

"""
python3 visualize.py model_path output_path
"""

model_path = sys.argv[1]
model = load_model(model_path)

output_path = sys.argv[2]

num_im = 20

# Open the hdf5 file
f = h5py.File('data.hdf5','r')

# Plot training images
train_index = np.random.randint(0,f['X_train'].shape[0]-1, num_im)
for i in train_index:
    # Initialize the subplots
    fig, axes = plt.subplots(nrows=1, ncols=3)
    # Plot the input data
    X = f['X_train'][i,:,:]
    axes[0].imshow(X)
    axes[0].set_title('X_train')
    # Plot the target data
    axes[1].imshow(f['Y_train'][i,:,:])
    axes[1].set_title('Y_train')
    # Plot the predicted image
    axes[2].imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:])
    axes[2].set_title("Prediction")
    # Add title and save the plot
    fig.suptitle("Training images: "+str(i))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(output_path+'/train_'+str(i))

# Plot validation images
valid_index = np.random.randint(0,f['X_valid'].shape[0]-1, num_im)
for i in valid_index:
    # Initialize the subplots
    fig, axes = plt.subplots(nrows=1, ncols=3)
    # Plot the input data
    X = f['X_valid'][i,:,:]
    axes[0].imshow(X)
    axes[0].set_title('X_valid')
    # Plot the target data
    axes[1].imshow(f['Y_valid'][i,:,:])
    axes[1].set_title('Y_valid')
    # Plot the predicted image
    axes[2].imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:])
    axes[2].set_title("Prediction")
    # Add title and save the plot
    fig.suptitle("Validation images: "+str(i))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(output_path+'/valid_'+str(i))

# Plot test images
test_index = np.random.randint(0,f['X_test'].shape[0]-1, num_im)
for i in valid_index:
    # Initialize the subplots
    fig, axes = plt.subplots(nrows=1, ncols=3)
    # Plot the input data
    X = f['X_test'][i,:,:]
    axes[0].imshow(X)
    axes[0].set_title('X_test')
    # Plot the target data
    axes[1].imshow(f['Y_test'][i,:,:])
    axes[1].set_title('Y_test')
    # Plot the predicted image
    axes[2].imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:])
    axes[2].set_title("Prediction")
    # Add title and save the plot
    fig.suptitle("Test images: "+str(i))
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.savefig(output_path+'/test_'+str(i))


f.close()