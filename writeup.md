
# Traffic Sign Recognition

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic signs data set listed as: 

The size of training set is 34799;
The size of the validation set is 4410;
The size of test set is 12630;
The shape of a traffic sign image is (32,32,3)
The number of unique classes/labels in the data set is 43.


#### 2. Include an exploratory visualization of the dataset.

Here is a random picked data and an exploratory visualization of the data set. It is a bar chart showing how the data distrubuted along different class.

1: Randomly picked data

![_auto_0](attachment:_auto_0)

2: Statistic analysis of dataset distribution

![_auto_0](attachment:_auto_0)

### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

* The first step is to convert image to greyscale because it helps to improve differentiation degree. 
* The second step is to normalize the dataset so the data has mean zero and equal variance. 
* Additional step could be augmentation, but because of time limit I decide to go with original dataset. The result shows relatively good. 

2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The final model archtecture is convolutional neural network. I followed LeNet to form a similar structure:

* Input: 32x32x1 image
* Layer 1: Convolution with 5x5 kernel, stride of 1, depth of 6, valid padding
*          Max pooling with 2x2 kernel, stride of 2, valid padding
* Layer 2: Convolution with 5x5 kernel, stride of 1, depth of 16, valid padding
*          Max pooling with 2x2 kernel, stride of 2, valid padding
* Flatten
* Fully Connected Layer with 120 hidden unit
* Dropout
* Fully Connected Layer with 84 hidden units
* Dropout
* Fully Connected Layer with 43 hidden units, The final 43 logits corresponding to target classed

3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I trained my model use typical Gradient Descent algorithem. 

* The original number of epochs is 10, but it didn't converge to the expected result, then I increased the number of epochs to 20, 30 and 40, test them seperately. It turns out with current setup, the number of epochs equals to 30 gives best accuracy with better efficience.

* The batch size is set to 128 is because too small batch size will calculate significant variance and cause slow convergence. Too large batch size will take more more time to calculate and may bring similar accuracy with my regular batch size.

* The hyperparameters were chosen is because a random hyperparameter combination search. 

4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

Validation set accuracy of 0.957
Test set accuracy of 0.937

If a well known architecture was chosen:

* What architecture was chosen?
** The original architecture is based on LeNet. In my architecture, with LeNet as a base, a convolutional neural network is chosen with supervised learning practices. 

* Why did you believe it would be relevant to the traffic sign application?
** LeNet project develops same image classification algorithem with distinguished performance, then I decided to go with LeNet on my traffic sign classification project.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
** After 30 iterations, the validation accuracy is about 0.957, and the test accuracy is 0.937 which provides good accuracy. Although there are lots of improvement, those will be my further project to implement.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

![_auto_1](attachment:_auto_1)

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Loaded images from disk, then preprocess those image use same workflow in previous discussed.  The prediction result "predictions_statics" stands for the predicted propability for each image correlated with different traffic signs.  

#### Predictions on additional sample images

* sampleimage/stop.ppm ==> Stop
 
* sampleimage/speed_limit_80.ppm ==> Speed limit (80km/h)
 
* sampleimage/caution.ppm ==> General caution
 
* sampleimage/right_lane_only.ppm ==> Keep right
 
* sampleimage/go_straight_or_left.ppm ==> Go straight or left

3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

Since those images I chosen have very clear characteristics, the propability results look very good with the correct prediction is almost 1. It may be too good to be true, but future test will be perform with other images with less quality and more noise.

Such as in the first image, the result looks that: 
[  1.00e+00,   2.00e-10,   1.31e-10,   3.67e-12,   6.14e-14], ==> [14, 17,  3, 33, 38]

14 stands for "Stop".



```python

```
