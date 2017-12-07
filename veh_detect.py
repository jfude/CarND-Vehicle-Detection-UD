import glob
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import time
import collections
from scipy.ndimage.measurements import label
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.model_selection  import GridSearchCV
from sklearn.svm import SVC
import sys
import pickle



## Convert img from one color space to another
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
    if conv == 'BGR2RGB':
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if conv == 'RGB2BGR':
        return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    

## Define a function to compute binned color features                                                    
def bin_spatial(img, spatial_size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector                                             
    # features = cv2.resize(img, size).ravel()
    # Return the feature vector                                                                         
    color1 = cv2.resize(img[:,:,0], spatial_size).ravel()
    color2 = cv2.resize(img[:,:,1], spatial_size).ravel()
    color3 = cv2.resize(img[:,:,2], spatial_size).ravel()
    return np.hstack((color1, color2, color3))


## Define a function to compute color histogram features                                                 
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately                                            
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector                                           
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms                                  
    return hist_features



## Get Histogram of Gradient features for an image
## Return hog feature vector and optionally image of features
## input
##      
##        pix_per_cell   -- pixels per cell
##        cell_per_block -- size of blocks traversing image
##        orient         -- number of possible gradient orientations per block         
##        vis            -- if True return feature image
##  
def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True                                                                
    if vis == True:
        features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell\
),
                                  cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys'\
, transform_sqrt=True,
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output                                                                    
    else:
        features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), block_norm='L2-Hys', transform_sqrt=True,
                       visualise=vis, feature_vector=feature_vec)
        return features



# Function to extract features from a list of images                                           
# This function is used for extracting features from the training images
# Returns the bin spatial and histogram color feature vectors and HOG feature vectors
# all concatenated together for each image.  
def extract_features(imgs, cspace='RGB', orient=9,
                     pix_per_cell=8, cell_per_block=2, hog_channel=0,
                     spatial_size=(32,32), hist_bins=32,hist_range=(0, 256)):
    
    
    ## input png images are on scale 0 -1 
    # Create a list to append feature vectors to                                                        
    features = []
    # Iterate through the list of images                                                                
    n = 0
    for file in imgs:
        # Read in each one by one                                                                       
        image = mpimg.imread(file) ## need to normalize??
        n +=1
        if((n % 100)==0):
            print("...read ",n," images.")
        # apply color conversion if other than 'RGB'
        # Replace with color_conversion function above !!
        if cspace != 'RGB':
            if cspace == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif cspace == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif cspace == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif cspace == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif cspace == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
            else: 
                feature_image = np.copy(image)

        
        
        # Returned feature image is on scale 0 - 1
        # Get color features
        # Apply bin_spatial() to get spatial color features               
        spatial_features = bin_spatial(feature_image, spatial_size=spatial_size)
        # Apply color_hist() also with a color space option now
        hist_features = color_hist(feature_image, nbins=hist_bins, bins_range=hist_range)
            
        
        # Call get_hog_features() with vis=False, feature_vec=True                                      
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],
                                    orient, pix_per_cell, cell_per_block,
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
             
        # Append the new feature vector to the features list 
        features.append(np.concatenate((spatial_features, hist_features, hog_features)))
    # Return list of feature vectors                    
    return features




## Returns list of windows that progressively get smaller as we move up in the
## image (as x gets smaller) according to xlimit_p
## Window coordinates are in terms of cells, converted from pixel space using 
## pixels per cell
##
## e.g. input
##
##
##   xlimit_p         = (700,390), (near bottom of image,near horizon middle of image)
##   ylimit_p         = (30, 1260) (near left edge of image, near right of image)
##   initial_size_p   = 225        initial window edge lenght at xlimit_p[0] in pixel space 
##   num_yoverlap     = 3          number of horizontal window overlaps allowed at an x level
##   size_frac        = 0.86       Percentage by which to reduce the size of the window edge     
#                                  when moving from x level to the next x level.
##   vert_frac        = 0.269      Fraction of the current window size to move upwards in the image
#                                  when moving from x level to the next x level.

def sliding_windows(xlimit_p,ylimit_p,initial_size_p,num_yoverlap,size_frac,vert_frac,
                    pixels_per_cell):

    windows = []
    size = round(initial_size_p/pixels_per_cell) # num pixels on edge
    bottom_left = np.array([0,0],np.int32)
    top_right   = np.array([0,0],np.int32)
    
        
    xlimit = (round(xlimit_p[0]/pixels_per_cell),round(xlimit_p[1]/pixels_per_cell))
    ylimit = (round(ylimit_p[0]/pixels_per_cell),round(ylimit_p[1]/pixels_per_cell))

    bottom_left[0] = xlimit[0]
    bottom_left[1] = ylimit[0]
    top_right[0]   =  bottom_left[0] - size
    top_right[1]   =  bottom_left[1] + size
    
    xlevel         =  xlimit[0]

    

    while(top_right[0] > xlimit[1] ):
        size            = round(size_frac*size)
        xlevel         -= round(vert_frac*size)
        yshift          = round(size/num_yoverlap) 
        bottom_left[0]  = xlevel
        bottom_left[1]  = ylimit[0]
        top_right[0]    = bottom_left[0] - size
        top_right[1]    = bottom_left[1] + size
       
    
        while(top_right[1] < ylimit[1] - yshift - 1):
            
            bottom_left[1] += yshift
            top_right[1]    = bottom_left[1] + size
            
            bright= (top_right[1],   bottom_left[0])
            tleft = (bottom_left[1], top_right[0])
           
            windows.append((tleft,bright))

    return windows



## draw boxes on image defined by label map for 
## that image
def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the blue box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img




def get_classifier(loadFeat,loadSVC,colorspace,orient,pix_per_cell,
                   cell_per_block,hog_channel,spatial,histbins):

    
    if(loadSVC == False):
        if(loadFeat==False):
            # Load up images and compute features for training
            car_images      = glob.glob('./vehicles/GTI_Far/*.png')
            car_images     += glob.glob('./vehicles/GTI_Left/*.png')
            car_images     += glob.glob('./vehicles/GTI_MiddleClose/*.png')
            car_images     += glob.glob('./vehicles/GTI_Right/*.png')
            car_images     += glob.glob('./vehicles/KITTI_extracted/*.png')

            noncar_images   = glob.glob('./non-vehicles/Extras/*.png')
            noncar_images  += glob.glob('./non-vehicles/GTI/*.png')

            print("Num Car images = ", len(car_images))
            print("Num Noncar images = ", len(noncar_images))


            t = time.time()
            print("Extracting car features...")
            car_features = extract_features(car_images, cspace=colorspace, orient=orient,
                                pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                hog_channel=hog_channel,spatial_size=(spatial,spatial),
                                hist_bins=histbins,hist_range=(0,1.0))

            print("Done.")
            print("Extracting non car features...")
            notcar_features = extract_features(noncar_images, cspace=colorspace, orient=orient,
                                   pix_per_cell=pix_per_cell, cell_per_block=cell_per_block,
                                   hog_channel=hog_channel,spatial_size=(spatial,spatial),
                                   hist_bins=histbins,hist_range=(0,1.0))   
            print("Done.")
            print("car feat len = ",    len(car_features))
            print("notcar feat len = ", len(notcar_features))
            t2 = time.time()
            print(round(t2-t, 2), 'Seconds to extract HOG features...')

            print("writing features file...")
            with open( 'features.pkl','wb') as f:
                pickle.dump([car_features,notcar_features],f)
            print("Done.")
        else:
            print("reading features file...")
            with open( 'features.pkl','rb') as f:
                car_features,notcar_features = pickle.load(f)
            print("Done.")
    
    
        print("car feat len = ",     len(car_features))
        print("car feat len0 = ",    len(car_features[0])) 
        print("notcar feat len = ",  len(notcar_features))
        print("notcar feat len0 = ", len(notcar_features[0]))

                
        # Create an array stack of feature vectors                                      
        X = np.vstack((car_features, notcar_features)).astype(np.float64)

        # Fit a per-column scaler                                                       
        X_scaler = StandardScaler().fit(X)
        # Apply the scaler to X                                                         
        scaled_X = X_scaler.transform(X)

        # Define the labels vector                                                      
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    
        # Split up data into randomized training and test sets                          
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(
            scaled_X, y, test_size=0.2, random_state=rand_state)
        

        print(X_train.shape)
        print(y_train.shape)
        print('Using:',orient,'orientations',pix_per_cell,
              'pixels per cell and', cell_per_block,'cells per block')
        print('Feature vector length:', len(X_train[0]))

        svc = SVC(kernel='linear',C=100.0)
        
        print("Starting fit...")
        t=time.time()
        svc.fit(X_train, y_train)
        t2 = time.time()
        print(round(t2-t, 2), 'Seconds to train SVC...')

        print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
        
        ## write out
        print("writing svc file...")
        with open( 'svc.pkl','wb') as f:
            pickle.dump([svc,X_scaler],f)
        print("Done.")
    else:
        print("reading svc file...")
        with open( 'svc.pkl','rb') as f:
            svc,X_scaler = pickle.load(f)
        print("Done.")
    
    
    return svc,X_scaler






#########################################
# Main
########################################

if __name__ == "__main__":
    

    ## Relevant parameters for feature extraction
    colorspace = 'YCrCb'                       
    orient = 9
    pix_per_cell = 8
    cell_per_block = 2
    hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"                                  
    spatial = 32
    histbins = 48

    
    
    ## Booleans so that previously calculated feature vectors or 
    ## previously trained may be quickly pulled from pickle file
    loadFeat = False
    loadSVC  = True

    svc,X_scaler = get_classifier(loadFeat,loadSVC,colorspace,orient,pix_per_cell,
                                  cell_per_block,hog_channel,spatial,histbins)
    
        

    ####### Initialize deque for heat map   #################
    ## qSize Length of deque
    qSize = 6
    q_heat_map = collections.deque(np.array((720,1280), dtype='int32'),qSize)
    
    #  Blank heat map
    blank = np.zeros((720,1280),'int32')
    #  
    for i in range(qSize):
        q_heat_map.append(np.copy(blank))

    ## Heat maps for iteration    
    heat_map = np.zeros((720,1280),'int32')
    current_heat_map = np.zeros((720,1280),'int32')
    first_heat_map = np.zeros((720,1280),'int32')
    final_heat_map = np.zeros((720,1280),'int32')
    
    ## Set a threshold for the summed q_heat_map
    heat_thresh = 10
    ##########################################################


    ##########################################################
    ## xybounds on the image
    xlimit         = (700,390) ## bottom to top
    ylimit         = (30, 1260) 
    ## create sliding window set
    ## 
    initial_size   =  225
    size_frac      =  0.86
    num_yoverlap   =  3
    vert_frac      =  0.269
    print("Getting windows...")
    windows = []
    windows = sliding_windows(xlimit,ylimit,initial_size,num_yoverlap,size_frac,vert_frac,pix_per_cell)
    print("num windows = ",len(windows))
    ##########################################################
    
    
    ### Load video ###########################################
    video = cv2.VideoCapture('../project_video.mp4')
    rval = video.isOpened()
    if(rval == False):
        print("Video failed to open.")
        sys.exit(0)
    ##########################################################
    

    ##########################################################
    # Open output video
    frameStart = 0
    frameStop  = 1280
    for i in range(frameStart):
        rval, frame = video.read()
        if(rval==False):
            print("Failed to read frame.")
            sys.exit(0)
    
    ### Sorensen video 3 codec is used for video output  
    fourcc = cv2.VideoWriter_fourcc('S', 'V', 'Q', '3')
    video_writer = cv2.VideoWriter('./project_output.sv3',fourcc,20.0,(1280,720),True)
    if(video_writer.isOpened() == False):
        print("video_writer failed to open")
        sys.exit(0)
    ##########################################################


    ## Calculate number of blocks traversed over training image 
    ## based on image size number of cells per block , training image
    ## is just hard set here, but could be returned from get_classifier
    ## with small mod
    twin_size = 64
    nb_trav_win  = (twin_size//pix_per_cell) - cell_per_block +1
    
    print("nb = ",nb_trav_win)
    

    ## Main loop over video frames
    ##
    fCount = frameStart
    while(fCount < frameStop):
        fCount +=1
        print("Reading frame #",fCount)   
        rval, frame = video.read()
        if(rval==False):
            print("Failed to read frame.")
            break

        
        
        img_orig = convert_color(frame,"BGR2RGB")
        img      = np.copy(img_orig)
        img_ctrans_tosearch  = convert_color(img, conv=("RGB2"+colorspace))

        # scale pixels from 0 - 255 to 0 - 1
        img_nctrans_tosearch = img_ctrans_tosearch.astype(np.float32)/255   
        
        ## Get all features for the window 
        hog1 = get_hog_features(img_nctrans_tosearch[:,:,0], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog2 = get_hog_features(img_nctrans_tosearch[:,:,1], orient, pix_per_cell, cell_per_block, feature_vec=False)
        hog3 = get_hog_features(img_nctrans_tosearch[:,:,2], orient, pix_per_cell, cell_per_block, feature_vec=False)


    
    
        ## Blank heat map for this new frame
        heat_map[:,:] = blank[:,:]
        ## Loop search windows for this frame
        for window in windows:
    
            ypos = window[0][0] ## positions are in terms of cells
            xpos = window[0][1]
    
            hog_feat1 = hog1[xpos:xpos+nb_trav_win, ypos:ypos+nb_trav_win].ravel()
            hog_feat2 = hog2[xpos:xpos+nb_trav_win, ypos:ypos+nb_trav_win].ravel()
            hog_feat3 = hog3[xpos:xpos+nb_trav_win, ypos:ypos+nb_trav_win].ravel()
            
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3)) ## same as concatenate
    

            xleft  = xpos*pix_per_cell
            ytop   = ypos*pix_per_cell
            xleft1 = window[1][1]*pix_per_cell
            ytop1  = window[1][0]*pix_per_cell
    
            ## Scale the windowed portion of the processed frame to the training image size
            ##
            subimg = cv2.resize(img_nctrans_tosearch[xleft:xleft1,ytop:ytop1], (twin_size,twin_size))
    
    
            ## Get Color features for this window
            spatial_features = bin_spatial(subimg, spatial_size=(spatial,spatial))
            ## Note pixels are 0 to 1 here
            hist_features    = color_hist(subimg,  nbins=histbins, bins_range=(0,1.0))

    
    

            # Scale features and make a prediction based on the extracted feature vectors
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))

            test_prediction = svc.predict(test_features)
        
    
            ##
            ## Draw this window onto imaage or update heat map
            ## 
            if test_prediction == 1:    
                w0 = (window[0][0]*pix_per_cell,window[0][1]*pix_per_cell)
                w1 = (window[1][0]*pix_per_cell,window[1][1]*pix_per_cell)
                #cv2.rectangle(img_orig,w0,w1,(255,0,0),2)
                # Update the heat map
                heat_map[xleft:xleft1,ytop:ytop1] += 1
        
        
        # Update q_heat_map and sum 
        np.copyto(first_heat_map,q_heat_map[0])
        current_heat_map[:,:] = current_heat_map[:,:] + (heat_map[:,:] - first_heat_map[:,:]) 
        q_heat_map.append(np.copy(heat_map))
        
        
        ## Threshold heat map
        final_heat_map[:,:] = current_heat_map[:,:]
        final_heat_map[final_heat_map <= heat_thresh] = 0
        
        ## Find high heat areas, make labels  
        labels = label(final_heat_map)
        
        
        ## Use labels for defining a bounding boxes and draw 
        ## the boxes on the original frame
        new_img = draw_labeled_bboxes(np.copy(img_orig), labels)
        new_img2 = convert_color(new_img,"RGB2BGR")
        
        ## Write ouput to video
        video_writer.write(new_img2)


    ## Clean up
    video.release()
    video_writer.release()
    cv2.destroyAllWindows()

    print("Done.")





