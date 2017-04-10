## Udacity Self-Driving Car Engineer Nanodegree

### Subject: **Deep Learning**
### Project: **Behavioral Cloning**
### Student: **Stephen Humphrey * Email: <stephenhumphrey@gmail.com>**
### Submitted: April 2017, version 001

***
In this project, I studied deep learning and a little control theory as I built a Convolutional Neural Network (CNN) to clone human driving behavior in a closed-track driving simulator. My CNN was based strongly on the one which runs in a real-world self-driving car created by Nvidia (see [End-to-End Deep Learning for Self-Driving Cars](https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/) and their engineers' [detailed report](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)). I successfully met the basic requirements of this project, which were to train, validate, and test a model using Keras on TensorFlow, and to use the model to control autonomously a car in the supplied driving simulator around one loop of a simple test track.

This document is the final write-up for the project. There is also another document which is the complete code notebook where all of the analysis, training, and testing code actually resides. There are also a driver file and a model file discussed and linked below.

[//]: # (Image References)

[NvidiaModelVisualization]: https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-768x1095.png "Nvidia Model Visualization"
[writeup-left-center-right-sample]: ./Figures/writeup-left-center-right-sample.png "writeup-left-center-right-sample"
[writeup-left-right-shift-sample]: ./Figures/writeup-left-right-shift-sample.png "writeup-left-right-shift-sample"
[writeup-left-right-shift-sample-2]: ./Figures/writeup-left-right-shift-sample-2.png "writeup-left-right-shift-sample-2"
[writeup-gamma-sample]: ./Figures/writeup-gamma-sample.png "writeup-gamma-sample"
[writeup-shadow-sample]: ./Figures/writeup-shadow-sample.png "writeup-shadow-sample"
[Nvidia-cnn-architecture]: ./Figures/Nvidia-cnn-architecture-768x1095.png "Nvidia-cnn-architecture"


***
Goal: Build a Simple Autonomous Vehicle in a Simulator
---
Using only a single, forward-facing camera video stream from the front of a simulated car, drive a complete circuit of a simple track by using machine learning to "clone" the steering behavior of a human driver.

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.
* Summarize the results with a written report
    * [this document]


***
Rubric Points
---
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.


***
Rubric part 1: Files Submitted & Code Quality
---


#### 1.1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py - containing the script to create and train the model
    * my model was built using Keras on TensorFlow
    * the model was built in this included Jupyter Notebook: [CarND-P3-Behavioral-Cloning.ipynb](./CarND-P3-Behavioral-Cloning.ipynb)
* drive.py - for driving the car in autonomous mode
    * I modified the supplied drive.py file to add support for joysticks (only used during testing, not during the autonomous track runs shown below)
    * I also modified the supplied supplied PI Controller in this file to add a derivative term (and thus, making it a simple PID controller)
    * the modified driver is included: [drive.py](./drive.py)
* model.h5 - containing a trained convolution neural network
    * the code in the Jupyter Notebook above created a Keras model file which is loaded by the drive.py to run inferences on the video stream from the simulator car while it drives
    * those inferences are used by the drive.py driver to make control inputs back to the simulator car (for steering, throttle, and braking)
    * the model is included: [model.h5](./model.h5)
* video of a successful run around the track
    * links for videos of my model and driver controlling the simulator are at the end of this report, as both YouTube and direct-download files
* writeup report summarizing the results
    * this document


#### 1.2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

If you are unable to run the model in a simulator yourself, please see the videos at the bottom of this document for a demonstration of my model in operation.


#### 1.3. Submission code is usable and readable

The [Jupyter Notebook file](./CarND-P3-Behavioral-Cloning.ipynb) contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works. I discuss this notebook in detail below.



***
Rubric part 2: Model Architecture and Training Strategy
---


#### 2.1. An appropriate model architecture has been employed

My model consists of a convolution neural network designed almost identically to the Nvidia model described in [End to End Learning for Self-Driving Cars (PDF)](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf).

![Nvidia-cnn-architecture][Nvidia-cnn-architecture]

It starts with a Keras Lambda layer, to normalize the input image data, then five convolutional layers of various dimensions, followed by three flat fully-connected layers. I had abundant compute capacity available, so I enlarged the depths of the convolutional layers compared to the Nvidia model, just to experiment with how to operationally handle a network with more trainable parameters. (see my model in my [Jupyter Notebook](./CarND-P3-Behavioral-Cloning.ipynb) in cell 38).

My model includes PReLU layers to introduce nonlinearity (code cell 38) because I was intrigued by the idea of having slightly leaky, trainable activation layers, as discussed in [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/pdf/1502.01852v1.pdf).


#### 2.2. Attempts to reduce overfitting in the model

The model contains dropout layers (which I tuned with a dropout-keep percentage of 60%) in order to reduce overfitting (between the FC layers in Notebook cell 38). I also used L2 Regularization (discussed in [Data Science 101: Preventing Overfitting in Neural Networks](http://www.kdnuggets.com/2015/04/preventing-overfitting-neural-networks.html/2)) for the same purpose.

The model was trained and validated on multiple data sets to ensure that the model was not overfitting (Notebook cell 3 identifies the data files, cells 4-6 load the files, and cell 13 is an image pool class for managing the cached loaded images). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 2.3. Model parameter tuning

The model was optimized using an Adam optimizer, so the learning rate was not tuned manually (Notebook cell 38). Its accuracy was measured by reducing a Mean-Square Error (MSE) loss function. Since I used 16,362 original driving images, which were augmented by computer vision techniques (discussed below) to create 1,000,000 training images, I used the Keras fit_generator() function to feed the model during training, so that all one million images wouldn't have to be loaded and managed simultaneously.


#### 2.4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used only center-lane driving, without so-called "recovery" data, gathered manually by driving the simulator with a proper joystick through two laps in each direction (forward and reverse) over the simple test track (track 1).

I had also gathered similar data from the two complex tracks (the mountain track in the current simulator as well as the jungle track in the original beta simulator), but testing revealed that these data did not help the model learn well, probably because I had gathered too many samples which weren't properly centered on the more complex tracks.

For details about how I created the training data by augmenting the manually produced human driving samples, see the next section.



***
Rubric part 3: Model Architecture and Training Strategy
---

#### 3.1. Solution Design Approach

The overall strategy for deriving a model architecture was to have an image-centric model which would learn and then later make inferences about appropriate steering angles and throttle and brake positions based only on the visual input of forward-facing cameras in the simulated vehicle.

My first step was to use a convolution neural network model similar to the Nvidia project. I thought this model might be appropriate because it had already proven itself capable on a similar problem set.

In order to gauge how well the model was working, I let the Keras fit_generator() function split my image and steering angle data into training and validation sets using its default parameters. I purposely fed my model an extremely small initial data set in order to prove that it would overfit and that everything was working properly. Afterward, I fed it over one million augmented images from the hand-gathered data. By spending a lot of time augmenting the captured datasets, I found that my first model worked very-well right out of the gate.

Although the simulated car would often depart from the track initially, I found that the inferences the model were making were entirely appropriate for the situation the car was in. I ended up spending much more time tuning the control functions in the driver (drive.py), including using a PID controller to decouple the steering angle controls from the observed left-right track position error. Since my needs weren't excessive, I modified the simple PI controller in the supplied drive.py file to add a derivative term, and thus ended up with a simple but proper PID controller.

My PID controller helped to minimize oscillations in the behavior of the car as it corrected itself from perturbations (like curves). In the first few seconds of each of the track runs demonstrated in the videos below, you will see the car swing wildly left and right for a few seconds before it settles down. This happens as the PID controller quickly "learns" the proper coefficients to feed back from its update function, particularly the integral (the I in PID). With a proper PID controller instead of the simple one in drive.py, I would have bothered to tune the coefficients to excise this initial oscillation, but it dampens quickly and so is not worth belaboring (until we get to actual control theory later in the course).

To combat overfitting, I modified the Nvidia model with L2 Regularization and with Dropout layers, and this seemed to work exceedingly well. I never experienced any significant overfitting issues, even after 20 epochs with 50,000 new augmented images in each epoch (see the bottom of Notebook cell 38).

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. I've included videos of a few of those sample laps at the bottom of this report.


#### 3.2. Final Model Architecture

The final model architecture (Notebook cell 38) consisted of a convolution neural network with the following layers and layer sizes:

| Layer | Description | Regularizer | Activation | Dropout |
|:---:|:---:|:---:|:---:|:---:|
| Input | RGB images 80 x 220, 3 channels | | | |
| Lambda | normalize pixels from [0,255] to [-1.0,1.0] | | | |
| Conv 1 | 5 x 5 convolution layer with 32 filters and 2 x 2 stride | L2 | PReLU | |
| Conv 2 | 5 x 5 convolution layer with 48 filters and 2 x 2 stride | L2 | PReLU | |
| Conv 3 | 5 x 5 convolution layer with 64 filters and 2 x 2 stride | L2 | PReLU | |
| Conv 4 | 3 x 3 convolution layer with 128 filters and 1 x 1 stride | L2 | PReLU | |
| Conv 5 | 3 x 3 convolution layer with 128 filters and 1 x 1 stride | L2 | PReLU | |
| Flatten | flatten 2D layers to 1D fully-connected layer | | | |
| FC 1 | Fully-Connected layer with 128 nodes, dropout-keep 60% between FC layers | L2 | PRuLU | 60% keep |
| FC 2 | Fully-Connected layer with 64 nodes, dropout-keep 60% between FC layers | L2 | PRuLU | 60% keep |
| FC 3 | Fully-Connected layer with 32 nodes, dropout-keep 60% between FC layers | L2 | PRuLU | 60% keep |
| Regression | A single inferred lane-position error prediction | | | |

Note that I used more filters than the Nvidia team did. I had abundant compute capacity available (not more than Nvidia, of course, just more than most students have), so I enlarged the depths of the convolutional layers compared to the Nvidia model just as an exercise in training more parameters. Nvidia's convolutional layers had [24, 36, 48, 64, and 64] filters respectively, while I used [32, 48, 64, 128, and 128] instead. Similarly, their FC layers were [100, 50, and 10] deep, while I used [128, 64, and 32], before the common single-node regression layer at the top.

With those expansions, as well as the L2 and PReLU parameters, my network ended with 1,539,217 trainable weights and other parameters (not counting the hyperparameters like dropout, which I tuned manually).


#### 3.3. Creation of the Training Set & Training Process

To capture good human driving behavior for my network to clone, I first recorded two laps each on the simple track using center lane driving, in both the forward and reverse directions. Here is an example image of center-lane driving, with the steering-angle at zero (leading to a shift error of 0.0), with the captured views from the left, center, and right cameras:

![writeup-left-center-right-sample][writeup-left-center-right-sample]

I did not generate so-called "recovery" images, because I surmised they would not be reliable (moving _into_ the position where the vehicle was recovering would have been difficult to tease out from moving _out_ of the recovery position, and I feared this "bad data" would pollute the training).

Instead, I used the left and right images to generate "recovery" offsets in the data, such that the left camera image was treated as "the vehicle is farther left than it should be" and the right camera images was treated oppositely (see Notebook cell 20) The "shift" term in these images is normalized between [0.0,1.0], such that 0.0 is as far left as the frame could be offset left, 0.5 is the frame is not offset, and 1.0 is the frame is as far right as possible. The "error" term is measured in pixels, and is calculated from the captured steering-angle the human driver was commanding at the moment each frame was taken.

![writeup-left-right-shift-sample][writeup-left-right-shift-sample]
![writeup-left-right-shift-sample-2][writeup-left-right-shift-sample-2]

For augmentation, any one of the left, right, or center images above could be randomly shifted left or right by the [0.0,1.0] shift factor, and a new error factor would be calculated based on the random shift.

To further augment the data sat, I also randomly flipped images and angles around their vertical (left-right) axis, so as to counter the natural bias in the track, where most turns are to the left.

In order to minimize the affect of brightness variations in different areas of the tracks, I augmented the original images by randomizing their Gamma brightness levels.

![writeup-gamma-sample][writeup-gamma-sample]

Finally, on a random basis, I would occasionally (10% of the time) generate faux "shadows" of various intensities across parts of the image to give the model lots of examples of features it needed to learn to ignore.

![writeup-shadow-sample][writeup-shadow-sample]

All of these augmentations were generated on-demand, as the Keras fit_generator() function would callback for more data, so I could have in theory generated nearly infinite combinations of new samples.

I used this augmented data both for training and for validating the model. The validation set helped determine if the model was over- or under-fitting. I found the ideal number of epochs was 20, not because I found any over-fitting yet at that point, but because I was getting good-enough results with the model. There were 50,000 images in each epoch (although unlike traditional CNN training, the augmentation process described above meant that I could feed whole new batches of images for each epoch). In total, I trained my model on exactly 1,000,000 augmented samples taken from the original 16,362 captured frames (with left, center, and right camera images in each frame).

There were naturally more frames captured with the steering angle near zero (since I generated them by slowly driving around the track while try to keep centered in the lane). To keep the model from becoming arbitrarily biased toward driving straight just because it has seen more samples were I was driving straight, I binned all the captured frames into groups with approximately the same steering angle (see Notebook cell 8) and then had my generator choose images from these bins equally randomly (see Notebook cell 34). Therefore, it was about just as likely the model would be trained on a frame where I was driving straight as it was on frames where I was setting the steering-angle to about 0.01, 0.02, 0.03, etc.



***
Example Track Runs
---
Here is a video on YouTube demonstrating one autonomously-driven loop around the Simple test track:

[![Self-driving loop on Simple Track](http://stephenhumphrey.s3-website-us-west-2.amazonaws.com/CarND/YouTube-2dYnFJW2ULU.png)](https://youtu.be/2dYnFJW2ULU?t=5s "Udacity CarND Behavioral Cloning - Term 1 - Project 3")

Here is a higher resolution version of the above video on Amazon S3:
* [P3-SimpleTrack-13mph-full.mp4 (282MB)](http://stephenhumphrey.s3-website-us-west-2.amazonaws.com/CarND/P3-SimpleTrack-13mph-full.mp4)

The view in the above video is from above and behind the car, but our software has a much smaller video feed, taken as if out of the front windshield. This is what our driver software "sees":
* [P3-SimpleTrack-13mph-full-out.mp4 (26MB)](http://stephenhumphrey.s3-website-us-west-2.amazonaws.com/CarND/P3-SimpleTrack-13mph-full-out.mp4)

Here is another, similar run on the same track, but with half-sized videos for quicker download:
* [P3-SimpleTrack-13mph-small.mp4 (142MB)](http://stephenhumphrey.s3-website-us-west-2.amazonaws.com/CarND/P3-SimpleTrack-13mph-small.mp4)
* [P3-SimpleTrack-13mph-small-out.mp4 (27MB)](http://stephenhumphrey.s3-website-us-west-2.amazonaws.com/CarND/P3-SimpleTrack-13mph-small-out.mp4)

