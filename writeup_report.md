# **Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/Nvidia_CNN_architecture.png "Nvidia CNN architecture"
[image2]: ./images/center_2019_01_15_23_03_54_135.jpg "Driving track in center"
[image3]: ./images/center_2019_01_18_02_07_14_245.jpg "Recovery Video"
[image4]: ./images/center_2019_01_18_02_07_15_247.jpg "Recovery Image"
[image5]: ./images/center_2019_01_18_02_07_16_039.jpg "Recovery Image"
[image6]: ./images/center_2019_01_15_23_03_54_135.jpg "Normal Image"
[image7]: ./images/center_2019_01_15_23_03_54_135_flipped.jpg "Flipped Image"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* utils.py containing supporting functions for model.py
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py & utils.py files contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 5x5 filter sizes with depths between 24 and 48 and 2 3x3 filter sizes with depths of 64 (utils.py lines 18-24)

The model includes RELU layers as activation of Convolution layers to introduce nonlinearity (utils.py, code line 59), and the data is normalized in the model using a Keras lambda layer (utils.py, code line 53).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (utils.py lines 60, 64, 68, 72, 76, 81, 83).

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py code line 14). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (utils.py line 87).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.
I used 4-5 laps of images on track1 with 2 laps of reverse laps & 2 laps on track2 & 1 lap of reverse laps on track2. Also used udacity provided data. And added 1 lap on track1 of recovering car from going out of track. Varied data helped the model training on total of 1 epochs and trained two times.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use well known Nvidia CNN architecture & with variable data sets.
Also, to add preprocessing on images by reducing size of image to focus on road & normalizing the image.

My first step was to use a convolution neural network model similar to the architecture mentioned in Nvidia paper [End to End Learning for Self-Driving Cars](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). I thought this model might be appropriate because it has less parameters than Lenet 5 model, and still is proven to give good results as per the paper.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I used the data provided by udacity. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model so that at each layer I used Dropout, which helped the model to train harder and learn better.

Then I added more training/validation data set with variation in driving styles. I used all 3 camera inputs & also added augmented data by flipping the images and correcting the steering angles for each. To corrected steering angle by 0.2 from center image to left/right camera input.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track after bridge is passed on the track. To improve the driving behavior in these cases, I added 1 lap of recovery lap where car drifted towards edges and recovered by strong steering away from edges.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (utils.py lines 42-88 in get_model function) consisted of a convolution neural network with the following layers and layer sizes. Below information can be found using `model_summary.py` by loading model and displaying summary().

| Layer         | Layer Specs                                | Output Size |
|---------------|--------------------------------------------|-------------|
| Normalization | lambda x: x / 255 - 0.5                    | 160x320x3   |
| Cropping      | Cropping2D(cropping=((50, 20), (0, 0))     | 90x320x3    |
| Convolution   | 24, 5x5 kernels, 2x2 stride, valid padding | 43x158x24   |
| RELU          | Non-linearity                              | 43x158x24   |
| Dropout       | Probabilistic regularization (p=0.5)       | 43x158x24   |
| Convolution   | 36, 5x5 kernels, 2x2 stride, valid padding | 20x77x36    |
| RELU          | Non-linearity                              | 20x77x36    |
| Dropout       | Probabilistic regularization (p=0.4)       | 20x77x36    |
| Convolution   | 48, 5x5 kernels, 1x1 stride, valid padding | 8x37x48     |
| RELU          | Non-linearity                              | 8x37x48     |
| Dropout       | Probabilistic regularization (p=0.3)       | 8x37x48     |
| Convolution   | 64, 3x3 kernels, 1x1 stride, valid padding | 6x35x64     |
| RELU          | Non-linearity                              | 6x35x64     |
| Dropout       | Probabilistic regularization (p=0.2)       | 6x35x64     |
| Convolution   | 64, 3x3 kernels, 1x1 stride, valid padding | 4x33x64     |
| RELU          | Non-linearity                              | 4x33x64     |
| Dropout       | Probabilistic regularization (p=0.2)       | 4x33x64     |
| Flatten       | Convert to vector.                         | 8448        |
| Dense         | Fully connected layer. No regularization   | 100         |
| Dropout       | Probabilistic regularization (p=0.5)       | 100         |
| Dense         | Fully connected layer. No regularization   | 50          |
| Dropout       | Probabilistic regularization (p=0.3)       | 50          |
| Dense         | Fully connected layer. No regularization   | 10          |
| Dense         | Output prediction layer.                   | 1           |



Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drift away from edges. These images show what a recovery looks like starting from left edge to center :
[Watch Edge Recovery Video](https://raw.githubusercontent.com/rosarp/CarND-Behavioral-Cloning-P3/master/images/recovery.mp4)
![alt text][image3] ![alt text][image4] ![alt text][image5]


Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would add more training data & help with edge recovery. For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

After the collection process, I had 36811 number of data points. I then preprocessed this data by reducing size -> normalizing images -> adding augmented data by using left & right images along with center and flipping each image with steering angle correction.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2 as evidenced by high training & validation loss (0.04*) but constant gradual reduction in first epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
