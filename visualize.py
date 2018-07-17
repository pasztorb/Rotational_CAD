import sys
import matplotlib.pyplot as plt
from keras.models import load_model
import h5py
import numpy as np

"""
python3 visualize.py model_path output_path
"""

def plot_figures(f, x_title, y_title):
    # Plot training images
    train_index = np.random.randint(0,f[x_title].shape[0]-1, num_im)
    for i in train_index:
        # Initialize the subplots
        fig, axes = plt.subplots(nrows=1, ncols=3)
        # Plot the input data
        X = f[x_title][i,:,:]
        axes[0].imshow(X, cmap='Greys_r')
        axes[0].set_title(x_title)
        # Plot the target data
        axes[1].imshow(f[y_title][i,:,:], cmap='Greys_r')
        axes[1].set_title(y_title)
        # Plot the predicted image
        axes[2].imshow(model.predict(X[np.newaxis ,np.newaxis,:,:])[0,0,:,:], cmap='Greys_r')
        axes[2].set_title("Prediction")
        # Add title and save the plot
        fig.suptitle(x_title[2:] + " images: "+str(i), y=9)
        fig.tight_layout()
        plt.savefig('{}/{}_{}'.format(output_path,x_title[2:],i))
        plt.close()


if __name__ == '__main__':
    model_path = sys.argv[1]
    model = load_model(model_path)

    output_path = sys.argv[2]

    num_im = 20

    # Open the hdf5 file
    file = h5py.File('data.hdf5', 'r')

    # Plot the images
    plot_figures(file, 'X_train', 'Y_train')
    plot_figures(file, 'X_valid', 'Y_valid')
    plot_figures(file, 'X_test', 'Y_test')

    # Close the hdf5 file
    file.close()