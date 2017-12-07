
Udacity CarND-Vehicle-Detection-Project
=======================================================




This repository contains the completed CarND-Vehicle-Detection-Project, which demonstrates
the use of Support Vector Machine classification to detect cars and distinguish them from other
features on the road. Both color and gradient features are used for training the classifier. 
A window search is then applied to each frame image taken by the camera. The classifier makes a 
prediction in each window and false positives are removed by accepting only positive detections 
in the same region in multiple consecutive frames. 

The primary output of the project is the code veh_detect.py which loads the project video,
project_video.mp4, and outputs the same video with bounding boxes surrounding positive detections
of cars. The output video is  'project_output.sv3'. My generation of this output (run of veh_detect.py)
has been uploaded as 'project_output_final.sv3' (sv3 is the orginal format)  and 'project_output_final.mov'
(converted format).


## List of Files
- README.md          -- this file.
- VHD_WriteUp.md     -- Description (with examples) of techniques used in this project and final project
  		     	code.
- project_video.mp4  -- Input video used for testing lane detection.
- project_output_final.sv3 -- Output video which is the project_video.mp4 with vehicle detection added (\
SV3 format).
- project_output_final.mov -- The same video but in Quicktime Movie format.

- veh_detect.py    -- The primary code that trains the SVM classifier, reads the project video,
  		      performs vehicle detection on each frame, and outputs the video with bounding boxes
		      around vehicles detected.
 
- ./examples/ -- Directory of images used in the write up for illustrating some stages of processing.



## Installation

-  Relevant software for the project was installed via miniconda using the "carnd-term1"
environment. carnd-term1 contains the following packages
```
conda list -n carnd-term1
```
appnope                   0.1.0                    py35_0    conda-forge
asn1crypto                0.22.0                   py35_0    conda-forge
backports                 1.0                      py35_1    conda-forge
backports.functools_lru_cache 1.4                      py35_1    conda-forge
blas                      1.1                    openblas    conda-forge
bleach                    2.0.0                    py35_0    conda-forge
ca-certificates           2017.7.27.1                   0    conda-forge
certifi                   2017.7.27.1              py35_0    conda-forge
cffi                      1.10.0                   py35_0    conda-forge
click                     6.7                      py35_0    conda-forge
cryptography              2.0.3                    py35_0    conda-forge
cycler                    0.10.0                   py35_0    conda-forge
dask-core                 0.15.4                     py_0    conda-forge
decorator                 4.0.11                    <pip>
decorator                 4.1.2                    py35_0    conda-forge
entrypoints               0.2.3                    py35_1    conda-forge
eventlet                  0.21.0                   py35_0    conda-forge
ffmpeg                    2.8.6                         0    menpo
flask                     0.12.2                   py35_0    conda-forge
flask-socketio            2.9.2                    py35_0    conda-forge
freetype                  2.7                           1    conda-forge
greenlet                  0.4.12                   py35_0    conda-forge
h5py                      2.7.1                    py35_1    conda-forge
hdf5                      1.8.18                        1    conda-forge
html5lib                  0.999999999              py35_0    conda-forge
idna                      2.6                      py35_1    conda-forge
imageio                   2.1.2                    py35_0    conda-forge
ipykernel                 4.6.1                    py35_0    conda-forge
ipython                   6.2.1                    py35_0    conda-forge
ipython_genutils          0.2.0                    py35_0    conda-forge
ipywidgets                5.1.5                    py35_0    menpo
itsdangerous              0.24                       py_2    conda-forge
jedi                      0.10.2                   py35_0    conda-forge
jinja2                    2.9.6                    py35_0    conda-forge
jpeg                      9b                            1    conda-forge
jsonschema                2.6.0                    py35_0    conda-forge
jupyter                   1.0.0                    py35_0    conda-forge
jupyter_client            5.1.0                    py35_0    conda-forge
jupyter_console           5.1.0                    py35_0    conda-forge
jupyter_core              4.3.0                    py35_0    conda-forge
Keras                     1.2.1                     <pip>
libffi                    3.2.1                         3    conda-forge
libgfortran               3.0.0                         0    conda-forge
libpng                    1.6.28                        1    conda-forge
libsodium                 1.0.10                        0    conda-forge
libtiff                   4.0.7                         0    conda-forge
markupsafe                1.0                      py35_0    conda-forge
matplotlib                2.1.0                    py35_0    conda-forge
mistune                   0.7.4                    py35_0    conda-forge
moviepy                   0.2.3.2                   <pip>
nbconvert                 5.3.1                      py_1    conda-forge
nbformat                  4.4.0                    py35_0    conda-forge
ncurses                   5.9                          10    conda-forge
networkx                  2.0                      py35_0    conda-forge
notebook                  5.1.0                    py35_0    conda-forge
numpy                     1.13.3          py35_blas_openblas_200  [blas_openblas]  conda-forge
olefile                   0.44                     py35_0    conda-forge
openblas                  0.2.19                        2    conda-forge
opencv3                   3.1.0                    py35_0    menpo
openssl                   1.0.2l                        0    conda-forge
pandas                    0.20.3                   py35_1    conda-forge
pandoc                    1.19.2                        0    conda-forge
pandocfilters             1.4.1                    py35_0    conda-forge
patsy                     0.4.1                    py35_0    conda-forge
pexpect                   4.2.1                    py35_0    conda-forge
pickleshare               0.7.4                    py35_0    conda-forge
pillow                    4.3.0                    py35_0    conda-forge
pip                       9.0.1                    py35_0    conda-forge
prompt_toolkit            1.0.15                   py35_0    conda-forge
protobuf                  3.4.0                     <pip>
ptyprocess                0.5.2                    py35_0    conda-forge
pycparser                 2.18                     py35_0    conda-forge
pygments                  2.2.0                    py35_0    conda-forge
pyopenssl                 17.2.0                   py35_0    conda-forge
pyparsing                 2.2.0                    py35_0    conda-forge
pyqt                      4.11.4                   py35_2    menpo
python                    3.5.2                         5    conda-forge
python-dateutil           2.6.1                    py35_0    conda-forge
python-engineio           1.7.0                    py35_0    conda-forge
python-socketio           1.8.1                      py_0    conda-forge
pytz                      2017.2                   py35_0    conda-forge
pywavelets                0.5.2               np113py35_0    conda-forge
PyYAML                    3.12                      <pip>
pyzmq                     16.0.2                   py35_2    conda-forge
qt                        4.8.7                         4
qtconsole                 4.3.1                    py35_0    conda-forge
readline                  6.2                           0    conda-forge
scikit-image              0.13.0                   py35_2    conda-forge
scikit-learn              0.19.0          py35_blas_openblas_201  [blas_openblas]  conda-forge
scipy                     0.19.1          py35_blas_openblas_202  [blas_openblas]  conda-forge
seaborn                   0.8.1                    py35_0    conda-forge
setuptools                36.3.0                   py35_0    conda-forge
simplegeneric             0.8.1                    py35_0    conda-forge
sip                       4.18                     py35_1    conda-forge
six                       1.11.0                   py35_1    conda-forge
sqlite                    3.13.0                        1    conda-forge
statsmodels               0.8.0                    py35_0    conda-forge
tbb                       4.3_20141023                  0    menpo
tensorflow                0.12.1                    <pip>
terminado                 0.6                      py35_0    conda-forge
testpath                  0.3.1                    py35_0    conda-forge
Theano                    0.9.0                     <pip>
tk                        8.5.19                        2    conda-forge
toolz                     0.8.2                      py_2    conda-forge
tornado                   4.5.2                    py35_0    conda-forge
tqdm                      4.11.2                    <pip>
traitlets                 4.3.2                    py35_0    conda-forge
wcwidth                   0.1.7                    py35_0    conda-forge
webencodings              0.5                      py35_0    conda-forge
werkzeug                  0.12.2                     py_1    conda-forge
wheel                     0.30.0                     py_1    conda-forge
widgetsnbextension        1.2.3                    py35_1    menpo
xz                        5.2.3                         0    conda-forge
zeromq                    4.2.1                         1    conda-forge
zlib                      1.2.8                         3    conda-forge
