from keras.models import load_model
import h5py
import sys
import numpy as np

input_data_path = sys.argv[1]
model_path = sys.argv[2]

# Load data
with h5py.File(input_data_path, 'r') as f:
    X_test = f['X_test'][()]
    Y_test = f['Y_test'][()]

# Add the extra channel axis
X_test = X_test[:,np.newaxis,:,:]
Y_test = Y_test[:, np.newaxis, :, :]

# Load model

model = load_model(model_path)

print(model.evaluate(x=X_test, y = Y_test))