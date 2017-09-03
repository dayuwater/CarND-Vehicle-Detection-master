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

[image7]: ./output_images/all_sliding_windows.png
[image8]: ./output_images/detected_sliding_windows.png
[image9]: ./output_images/single_1.png
[image10]: ./output_images/single_2.png
[image11]: ./output_images/cumulative_1.png
[image12]: ./output_images/cumulative_2.png

[image13]: ./output_images/false_two_detections.png
[image14]: ./output_images/false_caused_by_shade.png
[video1]: ./output_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

All of my code are in `lesson_functions.py` and `project.ipynb`. The first file is used to store some helper functions so that it does not take up space on the Notebook, and most of that file are code from lesson 34. `project.ipynb` is my Jupyter Notebook for experiments, all of the outputs are produced from this file.

`output_video.mp4` is the video you are looking for.

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

The code is in the block under "Step 6: One-functionize" in `project.ipynb`. (Line 17 - 30) The sliding_window search implementation is in `lesson_functions.py` (Line 213 - 241). This is the code copied from lesson 34, it extracts the required features from a given section of an image, and use the classifier to predict whether the feature is a car or not. I used sliding window search in 2 scales. One is 64 * 64 window with an overlap of 0.7, the other one is 80 * 80 window with an overlap of 0.8. The smaller window is more effective in detecting farther cars, and the larger window is more effective in detecting closer cars. As you can see in the following images, the size of the bounding box fits the appeared size of the car, and this is how I decided the size of the sliding window. However, for the amount of overlapping, I just experimented with different value and I found the value stated above works best.

![alt text][image7]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

In order to improve the performance of my classifier, I implemented the sliding windows only to the region that cars are most likely to appear, like from 400 to 500 in y-axis. Because in test images and videos, the car position is at the left-most lane all the times. I also eliminated the left half of the image for searching area. Please see the last section of the writeup to see a more detailed discussion about this approach.


![alt text][image8]

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

I then store all the bounding boxes detected in one frame to a class here: This is two blocks before "Step 6: Apply head map on time domain ( consecutive frames )" in my `project.ipynb`
```python
class BoundingBoxes:
    def __init__(self, max_frame = 6):
        self.bboxes = []  # a list of recently discovered bounding boxes. This should be 2D
        self.max_frame = max_frame # maximum frames of bboxes to store
        
    def add_bboxes(self, bboxes):
        # if there are less than the maximum frames stored, add in
        if len(self.bboxes) < self.max_frame:
            self.bboxes.append(bboxes)
        # else, pop the first one
        else:
            self.bboxes.pop(0)
            self.bboxes.append(bboxes)
 ```
 This makes sure that I can use heatmap to threshold out those false detections that only occurs in some frames recently. In addition, this smooth out the variances of box positions in consecutive frames.

Here's an example result showing the heatmap from a series of frames of video and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image9]
![alt text][image10]


### Here are six frames and their cumulative heatmaps with a cumulative threshold of 6:

![alt text][image11]
![alt text][image12]






---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

- Although I used thresholding in both spacial and time domain, and this does reduces a lot of false positives. There are still some false positives. However, this must be dealt in another way because increasing the threshold value might be a overkill and introduces false negatives.
- Because for all of the test images and test videos provided for this project, the car is at the left most lane, which means it is completely unnessesary to detect vehicles to the left. I actually used this fact and eliminated all the left portion of the input image, and this does proves to be successful in eliminating false positives. However, if the car is not at the left most lane, this pipeline will fail to detect cars on the left. This does not mean hard thresholding in x-axis (left-right) is wrong. If we can combine this project with advanced lane finding project, and use the detected lane as the threshold point, this pipeline will work even if the car is not at the left most lane.
- When the car is at the right most position, the pipeline either does not work, or work in a strange way. See the following image: The pipeline considers this as two cars because it detects a back view and a rear view separately.
- Although I only used Y channel for HOG features, and this 5x the speed of the pipeline compared to use all YCrCb, but 2.5 frames per second is not ideal for speed. This is far from real time detection.

![alt text][image13]

- It cannot detect cars that are in a farther distance ( perhaps 100m away ). Perhaps a more sophisticated sliding window approach is required.

In order to further check the robustness of the pipeline, I used the two challenge videos from Advanced Lane detection project as well. I also found the following problems arise from those two videos:
 
- Although it can detect cars that are about 80m - 100m away, the size of the bounding box is not correct.
- It will picks up shades as cars. Especially the shades caused by bridges.

![alt text][image14]

- It does not pick up motorcycles. This could be solved by adding more training data for motorcycles and other types of vehicles, perhaps pedestrains as well.
- It does not work well on picking up cars driving in opposite direction using the same parameters for detecting cars driving in same direction. Althogh extra parameter tuning will work in some sense,( Note: Please tune this parameter when grading my code: This is in the last block of my Jupyter Notebook.) it does increases false positives. However, tuning this manually is not an ideal solution. It is better to detect those lanes and use different size of storages and thresholds for different driving directions.
```python
bb = BoundingBoxes(30) # use 10 for detecting opposite lane, 30 for lanes driving in the same direction
```




