These are updated instructions to run the code: 

# NIIComputerVision

This project use python 2.7

To run the code you need to install several libraries :
* opencv :  conda install -c conda-forge opencv 
* numpy :  conda install -c anaconda numpy 
* import PIL : PIP install image
* scipy : conda install scipy
* skinage.draw : conda install -c conda-forge scikit-image
* plyfile: conda install -c kayarre plyfile 
* sklearn decomposition : conda install scikit-learn
* pandas : conda install pandas
* pyquaternion :  pip install pyquaternion
* imp :  conda install -c salilab imp 


Using GPU:
1) Install GPU drivers (depend on your GPU). Not needed on MAC
2) Install OpenCL. (Not needed on MAC)
3) Pyopencl : This project uses pyopencl version 2017.2 Newer versions are not supported 
            You can install this version from https://pypi.org/project/pyopencl/2017.2/
            pip install pyopencl==2017.2

* Be sure to copy .icd files in your /etc/OpenCL/vendors to the environments vendors  ~/anaconda2/envs/ENVIRONMENTNAME/etc/OpenCL/vendors
 

To visualize mesh:

* MeshLab (software)
http://www.meshlab.net/

* mayavi (directly with the code but limited compare to MeshLab) :
conda install mayavi

File
Create these files under the same location where code and data files are located
* ./meshes contains all the .ply file that are created during the execution of the code
* ./lib contains all the .pu file that are module of the project
* ./images is used for displaying a menu.
* ./boundingboxes, ./segment conntrains all .png file which are created during the execution of the code

Dataset
Download and extract all the files under data/ folder
* https://drive.google.com/drive/folders/1-lND28OLWwttADJng0yILeDv2cWiu6Nh?usp=sharing

Running
on terminal run dynamicFusion.py under code/ folder 

Common Errors:
Other CUDA versions may lead some OpenCL-CUDA library confusion
*  version 'OPENCL_2.x' not found
There is a simple "solution": try to move away the one from cuda and check if 
it fixes the issue, like:
```
sudo mv /usr/local/cuda-10.0/targets/x86_64-linux/lib/libOpenCL.so.1  
/usr/local/cuda-10.0/targets/x86_64-linux/lib/libOpenCL.so_from_cuda_do_not_use
```
