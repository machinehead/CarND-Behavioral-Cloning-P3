#**Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[central_cam_angle_distribution]: ./examples/central_cam_angle_distribution.png "Distribution of raw angle values for all central camera images in the recorded data"
[raw_image_shadow]: ./examples/center_2017_11_06_10_32_42_423.jpg "Raw center camera image with a shadow"
[image_shadow_normalized]: ./examples/2017_11_06_10_32_42_423.jpg "Center camera image with a shadow after histogram equalization"
[center_lane_driving]: ./examples/center_2017_11_05_19_34_30_916.jpg "Center lane driving example"
[flipped_center_lane_driving]: ./examples/flipped_center_2017_11_05_19_34_30_916.jpg "Center lane driving example (flipped)"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image" 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* modelConf.json containing the values of the configurable parameters
* random_search.py containing a driver script to perform random search for model parameters
* random_search_results.ipynb - a Jupyter notebook containing analysis of the random search results
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 3 layers using 5x5 filter sizes and depths of 10, 14, 18 (train.py lines 62-71 and modelConf.json lines 2-7) 

The model includes RELU layers to introduce nonlinearity (all convolutional layers and all dense layers except the last one), and the data is normalized in the model using:
* a Keras lambda layer to implement a trapezoidal interest region selection (train.py line 58).
* a Keras lambda layer to normalize and center pixel values (train.py line 60). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (train.py lines 75-83). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (train.py line 35 obtains a train/validation split).
 
See the next section for a description of how the dropout layers were chosen.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (train.py line 87).

For tuning other parameters, I built a process for randomly exploring the parameter space and choosing the most important parameters.

First, I would run the random_search.py script for a while; this script generates a random modelConf.json file and runs train.py for training and validating a new model. For every run train.py outputs a log file containing the values of all parameters and training/validation losses.

Second, I would calculate the Spearman rank correlation coefficient (see random_search_results.ipynb) between the minimum validation loss and every model parameter individually.
 
Third, I would choose a parameter with the maximal correlation coefficient, fix the best value of that parameter and repeat starting from the previous step again.
 
To my surprise, through this procedure I've learned that the most important parameter is to disable the dropout layer before the second dense layer (the current results in random_search_results.ipynb don't really show that, since after figuring out a good value for some parameter, I would fix that value and continue running random search; this allows to reduce the search space a lot, but reduces the value of the rank correlation coefficient for that parameter later).

Other discoveries were that:
* MaxPooling layer was helpful; I had some doubts about it;
* 3 or 4 convolutional layers perform better than 5 or less than 3; for that reason the final model uses 3 layers;
* Second dense layer performed somewhat better when there were more neurons (dense_2_factor >= 5);
* Third dense layer was the opposite (dense_3_factor <= 5).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. 

I didn't like the idea of recording recovery driving, since this is something that can't easily be done in real-life scenarios; for that reason I hoped that side camera images would help. I figured out that for the first track recording just the center lane driving would be enough if side cameras are also used; however, for the second track this approach works worse, especially at sharper turns. 

Finally, I used recordings of:
* center lane driving from both tracks; 
* center lane driving through the sharpest turns of track 1;
* center lane driving through one particularly bad turn of track 2;
* recovery driving away from a couple of dangerous turns of track 2.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

##### Track 1

The overall strategy for deriving a model architecture was to try the simplest possible architecture resembling a simplified version of what was explained in the video lectures as well as the NVIDIA architecture, etc.

My first step was to use a convolution neural network model similar to the one I used for the traffic sign recognition project, using a bunch of convolutional layers followed by a bunch of dense layers.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that it included a dropout layer in front of every dense layer. This helped improve the validation scores.

At this point I had had several iterations of trying to see the car drive around track one, but the car would always drive off the track at some random point. Usually it happened when the car was supposed to make a turn, but was driving straight instead.

I assumed the problem was in the distribution of the steering angles in the recorded data. Most of the recorded images have steering angle equal to 0: 

![Distribution of raw angle values for all central camera images in the recorded data][central_cam_angle_distribution]

I spent quite a lot of time to figure out where the problem was, but then I noticed someone on the Slack channel mentioned the problem of RGB <> BGR conversion. I figured out that my code was trained on BGR images and applied to RGB images while driving.
 
Right after I fixed this, the vehicle was immediately able to drive autonomously around track 1 without leaving the road.

##### Track 2

After the model was able to successfully drive through track 1, I decided to see how it would perform on track 2. It turned out that shadows on the road would present a major challenge for the model, since the signal to background ratio is much lower in those areas.
  
I knew that shadows could be addressed by artificially changing the brightness of the training images; however, I chose not to follow that route, because as I've learned from the traffic sign recognition project, augmenting the dataset in a way that's not present in real data leads to poorly performing models, and it's relatively hard to figure out how to properly augment the dataset for shadows.

Instead, I decided to see if histogram equalization applied to all the images would help fix the shadows problem. So I modified the training set generator function, as well as the drive.py script, to include OpenCV's histogram equalization.

Here's an image before and after histogram equalization (on the second image the green line indicates the correct steering angle, and the red line indicates the angle predicted by the model - image produced by the draw_angle_overlay.py script):

![Raw center camera image with a shadow][raw_image_shadow]

![Center camera image with a shadow after histogram equalization][image_shadow_normalized]

After carefully downsampling the training dataset (will be explained in later sections), using the best parameters found by the random search procedure, and recording a few more laps on track 2, I was able to obtain a model that successfully completes both track 1 and track 2.

#### 2. Final Model Architecture

The final model architecture (train.py lines 54-85) consisted of a convolution neural network with the following layers and layer sizes:

* A cropping layer
* An interest region selection layer (lambda)
* A normalization layer (lambda)
* 3 convolutional layers using 5x5 filter sizes and depths of 10, 14, 18 and RELU activations
* A MaxPooling layer with default parameters
* A 50% dropout layer
* A dense layer with 216 neurons and RELU activation
* A dense layer with 48 neurons and RELU activation
* A dense layer with 42 neurons and RELU activation
* A single output neuron

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center lane driving example][center_lane_driving]

I then recorded driving through the sharpest turns of track 1 in order to have more examples of higher steering angles.

To augment the data set, I also flipped images and angles thinking that this would make the dataset more symmetrical. On track 1, for instance, most of the turns are left turns, but we would still like the model to be able to turn right.

For example, here is an image that has then been flipped:

![Center lane driving example][center_lane_driving]
![Center lane driving example (flipped)][flipped_center_lane_driving]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
