## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_not_car.png

[image2]: ./output_images/HOG-analysis-0.jpg
[image3]: ./output_images/HOG-analysis-1.jpg
[image4]: ./output_images/HOG-analysis-2.jpg
[image5]: ./output_images/HOG-analysis-3.jpg

[image6]: ./output_images/color_features.png

[image7]: ./examples/HOG_example.jpg
[image8]: ./examples/sliding_windows.jpg
[image9]: ./examples/sliding_window.jpg
[image10]: ./examples/bboxes_and_heat.png

[image9]: ./examples/labels_map.png
[image9]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 2. Explain how (and identify where in your code) you extracted HOG features from the training images. Explain how you settled on your final choice of HOG parameters.

The code for this step is in lines 6 through 23 of the file called `lesson_functions.py`).  This code is from Lesson 34. I use this code because this function provides a wrapper that eliminates some insignificant parameters and enables conditional visualization.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed 10 images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

As you can see, the HOG features for the cars look similar, and the HOG features for non-cars are random. In addition, the HOG features for the cars and non-cars are mostly different. Therefore, the chosen parameter for HOG features are ideal for classification.

In addition, in order to increase the speed of classification and reduce the chance of overfitting. I examined the individual channels for YCrCb color space, and I found using Y channel solely is sufficient for classification because the color features for cars are similar, those for non-cars does not have a clear pattern, and cars and non-cars differ much. This proves to be crucial in later stage because this reduces false negatives.

![alt text][image6]



#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I first extracted the feature vector in `extract_features()` from Line 46 - Line 100 in `lesson_functions.py`. This code is from Lesson 34. I used spatial binning by resizing the image into 32 * 32, color histogram for all 3 color channels and HOG features for only Y channel together for classifier training. 

The machine learning code are in **Step 3 : Train the classifier** in `Project.ipynb`

I trained a linear SVM using the features extracted above. In order to make the data be scaled to zero mean and unit variance before training the classifier, I used a StandardScaler. 

```python
# Fit a per-column scaler
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X
scaled_X = X_scaler.transform(X)
```

I then use `train_test_split` to randomly split the train and test set without considering the effects of time series data. The size of the test size is 20%.

I then passed the scaled feature vectors and labels into a Linear SVC. The test accuracy is from 97% to 98%.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used sliding window search in 2 scales. One is 64 * 64 window with an overlap of 0.7, the other one is 80 * 80 window with an overlap of 0.8. The smaller window is more effective in detecting farther cars, and the larger window is more effective in detecting closer cars. As you can see in the following images, the size of the bounding box fits the appeared size of the car, and this is how I decided the size of the sliding window. However, for the amount of overlapping, I just experimented with different value and I found the value stated above works best.
![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to improve the performance of my classifier, I implemented the sliding windows only to the region that cars are most likely to appear, like from 400 to 500 in y-axis. Because in test images and videos, the car position is at the lest-most lane all the times. I also eliminated the left half of the image for searching area.


![alt text][image8]
![alt text][image9]
![alt text][image10]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

