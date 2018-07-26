# Rotational_CAD

This github repository was made for my project that concerns autoencoders.

## CAD.py
This is the main script that includes the models and their training. It can be run from terminal with the following command.
> python3 CAD.py model_type input_data_path output_path_to_directory

The first argument relates to the different types of models I considered. Possible types are: flat, convt, upsample, comb.

## preprocess_data.py
This script generates the training data from the MNIST dataset. Running it will generate the data into the working directory.

## visualize.py
This script plots the predicted images side-by-side with the input and the target images. Sample running command:
> python3 visualize.py path_to_model output_path_to_directory

Upon calling it will generate 20 images from the test, validation and training images each i.e. 60 overall.

## evaluation.ipynb
This is an iPython notebook I used to evaluate the test performance of the models and to examine the samples with the worst MSE results.

## test_evaluate.py
This script simply calculates the test error for the given model.
