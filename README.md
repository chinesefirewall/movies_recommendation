Movie Recommendation with TensorFlow Recommenders (TFRS)
=============================

#### This repository demonstrates how to build a simple movie recommendation model using the TensorFlow Recommenders (TFRS) library. The model takes movie titles and user IDs as inputs and recommends movies based on user preferences. The dataset used for this project is the movielens/100k-ratings and movielens/100k-movies from TensorFlow Datasets.  


## Table of content
- [Dependencies](*)
- [Data Loading](*)
- [Data Preprocessing](*)
- [Model Creation](*)
- [Full Model Implementation](*)
- [Model Training and Evaluation](*)
- [Recommendation and Model Serving](*)
- [Usage](*)
- [Dependencies](*)


##### The following libraries are used in the script:
os
pprint
Tempfile
typing: Dict, Text
Numpy as np
TensorFlow as tf
tensorflow_datasets as tfds
Data Loading  

The movielens/100k-ratings and movielens/100k-movies datasets are loaded using the tfds.load function. The ratings dataset contains user ratings for movies, while the movies dataset contains features of all available movies.  


##### Data Preprocessing
The ratings and movie datasets are preprocessed to extract relevant features. The ratings dataset is mapped to extract movie titles and user IDs, while the movie dataset is mapped to extract movie titles. The data is then shuffled and split into train and test sets.

##### Model Creation
Two embedding models are created for users and movies. These models are used to generate embeddings for user IDs and movie titles. The TFRS metrics and task are defined to measure the performance of the model and compute the loss, respectively.

##### Full Model Implementation
A custom TFRS model class, MovielensModel, is created by inheriting from tfrs.Model. This class takes user and movie models as inputs and implements the compute_loss method. The method computes the loss for given user and movie embeddings. Another custom model class, NoBaseClassMovielensModel, is created by inheriting from tf.keras.Model. This class defines the train_step and test_step methods to handle the training and testing of the model, respectively.

##### Model Training and Evaluation
The model is compiled using the Adagrad optimizer with a learning rate of 0.1. The train and test datasets are shuffled, batched, and cached to improve performance during training and evaluation. The model is then evaluated on the test dataset.

##### Recommendation and Model Serving
A BruteForce layer is used to create an index for movie recommendations based on user embeddings. The index is saved as a TensorFlow SavedModel format and can be loaded for serving the model. The model can also be deployed using TensorFlow Serving. Additionally, the ScaNN (Scalable Nearest Neighbors) layer is used to create an efficient index for movie recommendation. The index is saved and loaded in the same way as the BruteForce layer.

##### Usage
To use the movie recommendation model, simply load the saved index and pass a user ID as input. The model will return the top-recommended movie titles for the given user ID.



