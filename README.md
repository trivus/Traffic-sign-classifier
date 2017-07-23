#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./figures/figure0.png "Visualization"
[image2]: ./figures/figure1.png "preprocessing"
[image3]: ./figures/figure2.png "learning curve"
[image4]: ./figures/figure3.png "prediction on new images"
[image5]: ./figures/figure4.png "feature map of first conv layer"
[image6]: ./figures/figure5.png "feature map of second conv layer"
[image7]: ./figures/figure6.png "distribution map of samples"
[image8]: ./figures/figure7.png "new german signs"
[image9]: ./figures/figure8.png "original samples"
[image10]: ./figures/figure9.png "feature map conv1"
[image11]: ./figures/figure10.png "feature map conv2"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/trivus/Traffic-sign-classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used numpy library.

* The size of training set is 34799
* The size of the validation set is 4410.
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing distribution among classes.  
The data set is skewed, and three data set (i.e. train, valid, test) seems to follow similar distribution pattern. 

![Sample from training set][image1]
![Distribution plot of data sets][image7]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I applied preprocessing methods.  

1. normalization
2. grayscale, normalization
3. grayscale, histogram equalizer, normalization
4. grayscale, adaptive histogram equalizer, normalization
5. grayscale, image threshold, normalization

Here is an example of a traffic sign image before and after each methods.

![alt text][image9]
![alt text][image2]

I ran preliminary training test for each of methods above. I ran 5 iterations of 10 epochs training for each method.

Applying gray scaled helped to decrease training time with similar accuracy.

Equalizers did not improve accuracy significantly, but adaptive histogram method achieved slightly better accuracy on 3 iterations.

Image threshold negatively affected accuracy.  

Normalization was applied in all cases because dnn is known to work better with it. Simple normalization method was used: (x - 128)/128.

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grey scale image   							| 
| Convolution 5x5     		| 1x1 stride, valid padding, outputs 28x28x6, Relu 				|
| Max pooling	      		| 2x2 stride,  outputs 14x14x6 							|
| Convolution 5x5	        | 1x1 stride, valid padding, outputs 10x10x16, Relu				|
| Dropout		        | 										|
| Fully connected		| 400, 120        									|
| Dropout		        | 										|
| Fully connected		| 120, 84        									|
| Dropout		        | 										|
| logit				|      								|
					
 
I added three dropout layers to mitigate overfitting of the network.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

For final model, I used adam optimizer with learning rate of .0005, keep probability of 0.5 and 120 batch size, for 100 epochs.

![alt text][image3]

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of .996
* validation set accuracy of .974
* test set accuracy of .962

I first tested the LeNet provided in the previous lecture. There were strong signs of overfitting, so I added dropout layers.

I first used keep probability of 0.7, but the network completely overfitted to training data after 70 epochs. I lowered the keep probability to 0.5 for final model.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web, plus augmented version of three of them :

![alt text][image8]

I added three translated samples because the position of signs in the samples were off-center.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![alt text][image4]

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

After moving the signs to the center of the image, the classifier predicted all 5 samples correctly, where as it predicted only 2 correctly before the modification. The position of the sign seems to be a critical feature for the classifier. The training set included augmented samples but would benefit by adding more samples with extreme translation.


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

I compared feature maps of two stop sign images. The first image is an augmented image of the second one. THe first image was classified correctly, whereas the first image was not.
The feature maps of conv1 layer shows same patterns except for the position of the sign. As for the feature maps of conv2 layer, some of the maps do seems exhibit difference patterns, but it is unclear to what extent these maps contributed to the difference in prediction. However, featuremap of conv layer1 shows that the network is able to extract critical aspects of the signs, for example edge and letters.

![alt text][image5]
![alt text][image6]
![alt text][image10]
![alt text][image11]


