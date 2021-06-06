# Task 4 of Introduction to Machine Learning FS21

This repository contains two different approaches to this task:
1. using a transfer learning approach
2. using a siamese model with triplet loss

Both models passed baseline, however some improvements could be made:
- implement preprocessing for the images (random flip, crop etc.) for the transfer learning approach
- create disjoint train and validation sets (as it is, validation sets contain some of the already seen images, possibly fix 20% of range(5000) and create validation set from triplets including these images)
- siamese model runs extremely slowly, possible fix includes computing embeddings for all images and saving them to local drive, load them back during training, same thing for image augmentation and resizing)

## Sources:
https://keras.io/guides/transfer_learning/
https://keras.io/examples/vision/siamese_network/
https://github.com/yardenas/ethz-intro-ml/blob/master/project_4/cnns4food.py
https://github.com/giuliano-97/intro_ml/blob/master/task_4/main.py

## Explanation of the two approaches:
NOTE: Some parts have been commented out for reproducibility, for this notebook to work please change the paths for the files provided in the handout.

1) We first load the triplets into DataFrames and seperate them into three columns: 'anchor', 'positive', 'negative
2) We then unzip the food image into a new directory, resize our images to a size of (299,299) using tensorflow's resize_with_pad and subsequently save them into a directory food_res/img/. We had to use this additional folder for the keras image_dataset_from_directory function to work.
3) We then create an image dataset using said function, and use InceptionResNetV2 as our feature extractor, i.e. we create a base model (InceptionResNetV2) with pretrained weights for the ImageNet dataset, freeze all the layers using trainable = False, run our own image dataset we created through it to obtain the output of one of the layers (i.e. our features).
4) We then create a triplet tensor the following way: we label each row in the training dataframe with 1, and add the row with 'positive' and 'negative' switched with label 0, shuffle the whole frame using the sample method and then create an array row by row where for each triple in a row of the training dataframe we add the corresponding features. This way we obtain a train tensor and a list of labels.
5) We use these now as the input for our inference model
6) we then plot our training progress
7) Finally we create a test_tensor the same way we obtained the training tensor (but without shuffling this time), use our model to predict the labels and save them. Note that we had to delete our train tensor since we only had 12 GB of RAM available and would sometimes run into problems where we couldn't load both tensors and the model in our memory.

Our approach mostly follows the keras tutorial for siamese networks with triplet loss: https://keras.io/examples/vision/siamese_network/
Since we had some problems with passing the DataSets in above tutorial to our predict function we implemented a similar way of preprocessing the images as a group from a previous year: https://github.com/yardenas/ethz-intro-ml/blob/master/project_4/cnns4food.py
We also used their implementation of the loss function using softplus instead of taking a hard maximum of difference + margin and 0 since it seemed to perform better on the data.

1) We first load the triplets into DataFrames and use sklearn train_test_split to create a train and validation set from the training triplets
2) We then define functions which take a DataFrame containing triplets as inputs, create a tf.DataSets for each column and combine them with zip into 1 DataSet, apply our image loading + preprocessing functions which load an image, resize them to a target size (224,224) (we tried usign the maximum size for InceptionResNetV2 (299,299) but ran out of VRAM, and finally scale the values to [-1,1]. We then stack each image triplet into a tensor of size (3,224,224,3) and return it. We also return an additional '1' in the training case since the built in fit function of keras expects a target y even if you use a custom loss function. (We didn't really figure out how to implement a custom training loop)
3) Next we create our siamese network the following way: we use InceptionResNetV2 as our base cnn, freeze the top layer, use a pooling + dense layer and renormalize the data and use it to compute embeddings for each image in a triple, we then stack the three embeddings and return this
4) We then train our model for the training and validation data using a custom triplet loss function
5) As a last step we use an inference model which just calculates the distance between the two pairs of embeddings (anchor, positive) and (anchor, negative) and returns 1 if dist((anchor, negative), (anchor,positive)) <= 0 and 0 otherwise, i.e. 1 if Positive image is closer to the Anchor and 0 otherwise, and then use the predict function to run this on our test dataset. 