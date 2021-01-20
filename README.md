# Day-and-Night Image Classifier
This is an Image Classification of Self Drving Car Series.. We'd like to build a classifier that can accurately label these images as day or night, and that relies on finding distinguishing features between the two types of images!

![Img](https://github.com/jackyhuynh/Day_Night_Image_Classifier/blob/main/images/day_night.png)

# A note on neural networks
Neural networks to be a set of algorithms that can learn to recognize patterns in data and sort that data into groups. The example we gave was sorting yellow and blue seas shells into two groups based on their color and shape; a neural network could learn to separate these shells based on their different traits and in effect a neural network could learn how to draw a line between the two kinds of shells. Deep neural networks are similar, only they can draw multiple and more complex lines in the sand, so to speak. Deep neural networks layer separation layers on top of one another to separate complex data into groups.

![Img](https://github.com/jackyhuynh/Day_Night_Image_Classifier/blob/main/images/screen-shot-2018-01-16-at-5.55.04-pm.png)

# Convolutional Neural Networks (CNN's)
The type of deep neural network that is most powerful in image processing tasks is called a Convolutional Neural Network (CNN). CNN's consist of layers that process visual information. A CNN first takes in an input image and then passes it through these layers. There are a few different types of layers, but we'll touch on the ones we've been learning about (and even programming on our own!): convolution, and fully-connected layers.

## Convolutional layer
- A convolutional layer takes in an image array as input.
- A convolutional layer can be thought of as a set of image filters (which you've been learning about).
- Each filter extracts a specific kind of feature (like an edge).
- The output of a given convolutional layer is a set of feature maps, which are differently filtered versions of the input image.
- Fully-connected layer

![Img](https://github.com/jackyhuynh/Day_Night_Image_Classifier/blob/main/images/screen-shot-2018-01-16-at-6.01.18-pm.png)

## Classification from scratch (Day and Night example)
In this course, you've seen how to extract color and shape features from an image and you've seen how to use these features to classify any given image. In these examples, it's been up to you to decide what features/filters/etc. are useful and what classification method to use. You'll even be asked about learning from any classification errors you make.

This is very similar to how CNN's learn to recognize patterns in images: given a training set of images, CNN's look for distinguishing features in the images; they adjust the weights in the image filters that make up their convolutional layers, and adjust their final, fully-connected layer to accurately classify the training images (learning from any mistakes they make along the way). Building these layers from scratch, like you're doing, is a great way to learn the inner working of machine learning techniques, and you should be proud to have gotten this far!

## Further learning!
Typically, CNN's are made of many convolutional layers and even include other processing layers whose job is to standardize data or reduce its dimensionality (for a faster network). If you are interested in learning more about CNN's and the complex layers that they can use, I recommend looking at this article as a reference. 
- [An Intuitive Explanation of Convolutional Neural Networks](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)

# Evaluation Metrics

## Accuracy

The accuracy of a classification model is found by comparing predicted and true labels. For any given image, if the predicted_label matches thetrue_label, then this is a correctly classified image, if not, it is misclassified.

The accuracy is given by the number of correctly classified images divided by the total number of images. We’ll test this classification model on new images, this is called a test set of data.

## Test Data

Test data is previously unseen image data. The data you have seen, and that you used to help build a classifier is called training data, which we've been referring to. The idea in creating these two sets is to have one set that you can analyze and learn from (training), and one that you can get a sense of how your classifier might work in a real-world, general scenario. You could imagine going through each image in the training set and creating a classifier that can classify all of these training images correctly, but, you actually want to build a classifier that recognizes general patterns in data, so that when it is faced with a real-world scenario, it will still work!

So, we use a new, test set of data to see how a classification model might work in the real-world and to determine the accuracy of the model.

## Misclassified Images

In this and most classification examples, there are a few misclassified images in the test set. To see how to improve, it’s useful to take a look at these misclassified images; look at what they were mistakenly labeled as and where your model fails. It will be up to you to look at these images and think about how to improve the classification model!

# Complete image classification pipeline!

There are a lot of material are used to built the image classification, from probability to classification! We learned how to manually program a classifier step-by-step. First by looking at the classification problem and your images, and planning out a complete approach to a solution.

The steps include pre-processing images so that they could be further analyzed in the same way, this included changing color spaces. Then we moved on to feature extraction, in which you decided on distinguishing traits in each class of image, and tried to isolate those features! Finally, you created a complete classifier that output a label or a class for a given image, and analyzed your classification model to see its accuracy!

![img](https://github.com/jackyhuynh/Day_Night_Image_Classifier/blob/main/images/screen-shot-2017-12-18-at-3.21.09-pm.png)

## Technology
- Python 
- Object Oriented Design
- Jupyter Notebook
- Data Visualization
- Machine Learning
- AI
- Localization
- Prediction
- Data Structures
- Open CV

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites
What things you need to install the software and how to install them
- Jupyter Notebook: If you want just test the code, simply go to google and search for jupiter notebook or another Python online IDE. The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text. 
- Anacoda Navigator: Install Anaconda Navigator if you want to develop data sciences using python or R. Anaconda Navigator is a desktop graphical user interface included in Anaconda that allows you to launch applications and easily manage conda packages, environments and channels without the need to use command line commands. 

### Installing

A step by step series of examples that tell you how to get a development enviroment running

* [Install Anacoda Navigator](https://docs.anaconda.com/anaconda/navigator/install/#:~:text=Installing%20Navigator%20Navigator%20is%20automatically%20installed%20when%20you,install%20anaconda-navigator.%20To%20start%20Navigator,%20see%20Getting%20Started.) - If you haven't downloaded and installed Anacoda Navigator yet, here's how to get started.
* [Jupyter Notebook](https://jupyter.org/try) - Click here to go to the online free Jupiter Notebook.

## Running the tests

Explain how to run the automated tests for this system:
- There is no download IDE need, all you need is download all the src to your machine and run it.
- Using Jupiter Notebook
- Navigate to the file Classification.ipynb
- hit:

```
Ctrl + Enter
```
- The notebook will execute in Markdown form and include some data visualization to show the actual performance of the image classifier.

## Deployment

day night classification is extremely helpful for .Idea for localization and/or self-driving car.
It turns out that using multiple sensors like radar and lidar at the same time, will give even better results. Using more than one type of sensor at once is called sensor fusion, which is used in Self-Driving Car applications.
Please refer to my notebook for better understanding.

## Built With

* [Jupyter Notebook](https://jupyter.org/try) 

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Truc Huynh** - *Initial work* - [TrucDev](https://github.com/jackyhuynh)

## Format
my README.md format was retrieved from
* **Billie Thompson** - *Initial work* - [PurpleBooth](https://github.com/PurpleBooth)
See also the list of [contributors](https://github.com/your/project/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Udacity.com for outstanding lesson
