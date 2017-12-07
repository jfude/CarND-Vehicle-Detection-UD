
## Vehicle Detection Project

The goal of this project is to demonstrate a method for detecting nearby vehicles on the road for 
a self-driving car. The processing steps that are covered in this project are the following.
  


* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier an SVM classifier
* Apply a color transform and append binned color features, as well as histograms of color, to the HOG feature vector. 
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run this processing pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[clean_det1_img]: ./examples/clean_detect1.png
[clean_det2_img]: ./examples/clean_detect2.png
[clean_det3_img]: ./examples/clean_detect3.png
[false_pos1_img]: ./examples/false_pos1.png
[false_pos2_img]: ./examples/false_pos2.png


[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## Rubric Points  Points

Here I will provide a reference to the sections below that address each individual rubric. The rubric 
points and descriptions for this project may be found [here](https://review.udacity.com/#!/rubrics/513/view).

- Feature Extraction
  - [Data Set](#data-set)
  - [Color Features](#color-features)
  - [HOG Features](#hog-features)
- [Classifier Training](#classifier-training)
- [Sliding Window Set Generation]
- [Detection Reliablity](#detection-reliability)
- [Output Video](#output-video)
- [Discussion](#discussion)

---

## Feature Extraction

### Data Set

The data set for training our classifier was the set of all 64x64 png images from the directories

```
./vehicles/GTI_Far/
./vehicles/GTI_Left/
./vehicles/GTI_MiddleClose/
./vehicles/GTI_Right/
./vehicles/KITTI_extracted/
./non-vehicles/Extras/
./non-vehicles/GTI/
```

The number of car images (8792)  and non-car images (8968) was roughly equal,  important for 
training reliable classifier.   


### Color Features 


After selecting an appropriate color space, color features are characterized and extracted in two different ways. 
The first method is to simply reduce the input training image resolution and extract all pixels for all color channels.
Each color channel for the image is flattened and all are then concatenated. This is implemented in the function bin_spatial.
The second method is to make a histogram for each channel where each bin represents the number of pixels in a given intensity 
range. This implemented in the function color_hist, which can be passed the number of bins and the bin range. The bin range should
be set appropriately for the underlying pixel type of the input range. All three histograms are then concatenated together to 
produce a single color hist feature vector. 

In order to select a good color space and appropriate parameters for these methods, we extracted features on the provided
small test set of vehicle and non-vehicle images, split the images into train and test sets, trained a linear SVC classifier, 
and compared accuracy scores for various parameter values. The two best color spaces were found to be 'YCrCb' and 'LUV', giving
scores in the 90 and above percentile range, where as other choices produced scores in the high 80s or less.
A resolution of 16x16 for the spatial feature and 48 bins for the color histogram feature produced the best results. Later however, 
when running against the project video it was found that reducing resolution to 16x16 produced too many false positive detections,
so I returned to 32x32.   


### HOG Features

For spatial gradient characteristics I used the hog (Histogram of Gradients) function from skimage.feature library.
This function is called in get_hog_features. Initially running again the small test image set, I found that extracting all
color channels and a higher number of orientations (12 rather than the default of 9)  gave higher accuracy scores.  block-norm 


The parameters I chose were


I did not get a better result by increasing the number cells per block, or block size. 

The returned HOG feature vectors for each color channel are concatenated together and this result is further
concatenated with the aforementioned color feature vectors to produce the final feature vector.

Once feature vectors for all images were extracted, they were normalized using

```python
# Fit a per-column scaler                                                       
X_scaler = StandardScaler().fit(X)
# Apply the scaler to X                                                         
scaled_X = X_scaler.transform(X)
```

where X here is the full stack of all feature vectors.


## Classifier Training

A support vector machine classifier was used for this project. I attempted using
GridSearchCV for optimizing the C and gamma parameters, but this seemed to take an inordinate
amount of time. I settled on using a linear SVC and tried a few different C parameters, using C=100 in
the end. The data set was split 80% training and 20% for test, in previous test and in the final code.   

```python
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)

svc = SVC(kernel='linear',C=100.0)        
```

When features are extracted from a window of a frame of video, the features are similarly normalized
before a prediction is made by the classifier. 


```python
 test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))
 test_prediction = svc.predict(test_features)
```


## Sliding Windows

The algorithm for creating a set of sliding windows to traverse an image frame and detect vehicles is implemented in the function
sliding_windows(). This function creates a set of windows which are the same size at the same y (vertical) level in the image
and get progressively smaller as we move upward (smaller y values) in the image. This make sense as vehicles that are lower in the image
(not near the horizon), will not be as small as those on the horizon. 

All windows are square and there coordinates are returned in terms of cells, not pixels. 

The input parameters are

```python
# Limits that bound the area where windows are created.
xlimit_p = 
ylimit_p = 

# Percentage by which to reduce the size of the window edge     
# when moving from x level to the next x level.
size_frac = 

# Fraction of the current window size to move upwards in the image
# when moving from x level to the next x level.

vert_frac = 
  
# Number of horizontal window overlaps allowed at an x level. 

num_yoverlap 
# Number of pixels per cell
pixel_per_cell= 8  
```

where "size" here refers to a side edge of a square window. Window sizes change geometrically as a function of y.
The parameter choices produce 301 windows for searching essentially the bottom half of each frame.
 

## Detection Reliability

In many frames of the project video, the above method does a pretty good job at detecting cars,
including detecting cars on the opposite side of the median.

[images]


In a number of frames however, false positive detections appear. 

[images]

To remove false positives, a heat map is constructed by combining detections from a number of consecutive frames, and thresholding
areas with multiple detections.

We used a double ended for this purpose. 

   

```python

# Initialize 
heat_thresh = 8 
qSize = 10 # size of deque
blank = np.zeros((720,1280),'int32')  # blank heat map
q_heat_map = collections.deque(np.array((720,1280), dtype='int32')  ,qSize)


# Loop frames

   # For this frame, loop search windows
   for window in windows

       # Update heat map for this frame
        heat_map[xleft:xleft1,ytop:ytop1] += 1


   # Update summed heat map
   first_heat_map      = q_heat_map[0]
   q_heat_map.append(heat_map)
   current_heat_map += (heat_map - first_heat_map) # This is the summed heat map

   # Apply threshold
   current_heat_map[current_heat_map <= heat_thresh] = 0
```

The heat map for the above image with false positives is shown below. With a threshold
level of 8,  the false positive does not appear. The summed heat map is used for drawing
the final bounding box in an area of "high heat" or high detection. The label() function from 
the scipy.ndimage.measurements library was used in the following way 


```python
from scipy.ndimage.measurements import label
labels = label(current_heat_map)
final_img = draw_labeled_bboxes(np.copy(img_orig), labels)
```

where img_orig is the original image. 

The final image with bounding boxes drawn around detected vehicles is shown below. 



## Output Video

   The output video produced by the above steps implemented in veh_detect.py and run against ./project_video.mp4
   is ./project_output.sv3. I also provided this video convert to Quicktime format, ./project_output.mov.

## Discussion
  
Windows parameters
hog features over whole image
better optimize classifier.



