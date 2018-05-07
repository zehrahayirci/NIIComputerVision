"""
 File created by Diego Thomas the 16-11-2016
 improved by Inoe Andre from 02-2017

 Define functions to manipulate RGB-D data
"""
import cv2
import numpy as np
from numpy import linalg as LA
import random
import imp
import time
import scipy.ndimage.measurements as spm
import pdb
from skimage import img_as_ubyte
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import copy
from skimage.draw import line_aa

segm = imp.load_source('segmentation', './lib/segmentation.py')
General = imp.load_source('General', './lib/General.py')



class RGBD():
    """
    Class to handle any processing on depth image and the image breed from the depth image
    """

    def __init__(self, depthname, colorname, intrinsic, fact):
        """
        Constructor
        :param depthname: path to a depth image
        :param colorname: path to a RGBD image
        :param intrinsic: matrix with calibration parameters
        :param fact: factor for converting pixel value to meter or conversely
        """
        self.depthname = depthname # useless
        self.colorname = colorname # useless
        self.intrinsic = intrinsic
        self.fact = fact

    def LoadMat(self, Images,Pos_2D,BodyConnection, ColorImg):
        """
        Load information in datasets into the RGBD object
        :param Images: List of depth images put in function of time
        :param Pos_2D: List of junctions position for each depth image
        :param BodyConnection: list of doublons that contains the number of pose that represent adjacent body parts
        :return:  none
        """
        self.lImages = Images
        self.CImages = ColorImg
        self.hasColor = True
        if self.CImages.shape[0]==0:
            self.hasColor = False
        self.numbImages = len(self.lImages.transpose()) # useless
        self.Index = -1
        self.pos2d = Pos_2D
        self.connection = BodyConnection

    def ReadFromDisk(self):
        """
        Read an RGB-D image from the disk
        :return: none
        """
        print(self.depthname)
        self.depth_in = cv2.imread(self.depthname, -1)
        self.color_image = cv2.imread(self.colorname, -1)

        self.Size = self.depth_in.shape
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]):
                self.depth_image[i,j] = float(self.depth_in[i,j][0]) / self.fact

    def ReadFromMat(self, idx = -1):
        """
        Read an RGB-D image from matrix (dataset)
        :param idx: number of the
        :return:
        """
        if (idx == -1):
            self.Index = self.Index + 1
        else:
            self.Index = idx

        depth_in = self.lImages[0][self.Index]
        print "Input depth image is of size: " + str(depth_in.shape)
        size_depth = depth_in.shape
        self.Size = (size_depth[0], size_depth[1], 3)
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        self.depth_image_ori = depth_in
        self.depth_image = depth_in.astype(np.float32) / self.fact
        # self.skel = self.depth_image.copy() # useless

        # handle positions which are out of boundary
        self.pos2d[0,idx][:,0] = (np.maximum(0, self.pos2d[0, idx][:,0]))
        self.pos2d[0,idx][:,1] = (np.maximum(0, self.pos2d[0, idx][:,1]))
        self.pos2d[0,idx][:,0] = (np.minimum(self.Size[1], self.pos2d[0, idx][:,0]))
        self.pos2d[0,idx][:,1] = (np.minimum(self.Size[0], self.pos2d[0, idx][:,1]))

        # get kmeans of image
        if self.hasColor:
            self.color_image = self.CImages[0][self.Index]

    #####################################################################
    ################### Map Conversion Functions #######################
    #####################################################################

    def Vmap(self):
        """
        Create the vertex image from the depth image and intrinsic matrice
        :return: none
        """
        self.Vtx = np.zeros(self.Size, np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]): # column index (i.e. horizontal x axis)
                d = self.depth_image[i,j]
                if d > 0.0:
                    x = d*(j - self.intrinsic[0,2])/self.intrinsic[0,0]
                    y = d*(i - self.intrinsic[1,2])/self.intrinsic[1,1]
                    self.Vtx[i,j] = (x, y, d)


    def Vmap_optimize(self):
        """
        Create the vertex image from the depth image and intrinsic matrice
        :return: none
        """
        #self.Vtx = np.zeros(self.Size, np.float32)
        #matrix containing depth value of all pixel
        d = self.depth_image[0:self.Size[0]][0:self.Size[1]]
        d_pos = d * (d > 0.0)
        # create matrix that contains index values
        x_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        y_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        # change the matrix so that the first row is on all rows for x respectively colunm for y.
        x_raw[0:-1,:] = ( np.arange(self.Size[1]) - self.intrinsic[0,2])/self.intrinsic[0,0]
        y_raw[:,0:-1] = np.tile( ( np.arange(self.Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()
        # multiply point by point d_pos and raw matrices
        x = d_pos * x_raw
        y = d_pos * y_raw
        self.Vtx = np.dstack((x, y,d))
        return self.Vtx

    def NMap(self):
        """
        Compute normal map
        :return: none
        """
        self.Nmls = np.zeros(self.Size, np.float32)
        for i in range(1,self.Size[0]-1):
            for j in range(1, self.Size[1]-1):
                # normal for each direction
                nmle1 = General.normalized_cross_prod(self.Vtx[i+1, j]-self.Vtx[i, j], self.Vtx[i, j+1]-self.Vtx[i, j])
                nmle2 = General.normalized_cross_prod(self.Vtx[i, j+1]-self.Vtx[i, j], self.Vtx[i-1, j]-self.Vtx[i, j])
                nmle3 = General.normalized_cross_prod(self.Vtx[i-1, j]-self.Vtx[i, j], self.Vtx[i, j-1]-self.Vtx[i, j])
                nmle4 = General.normalized_cross_prod(self.Vtx[i, j-1]-self.Vtx[i, j], self.Vtx[i+1, j]-self.Vtx[i, j])
                nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
                # normalized
                if (LA.norm(nmle) > 0.0):
                    nmle = nmle/LA.norm(nmle)
                self.Nmls[i, j] = (nmle[0], nmle[1], nmle[2])

    def NMap_optimize(self):
        """
        Compute normal map, CPU optimize algo
        :return: none
        """
        self.Nmls = np.zeros(self.Size, np.float32)
        # matrix of normales for each direction
        nmle1 = General.normalized_cross_prod_optimize(self.Vtx[2:self.Size[0],1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1,2:self.Size[1]] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle2 = General.normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[0:self.Size[0]-2,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle3 = General.normalized_cross_prod_optimize(self.Vtx[0:self.Size[0]-2,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle4 = General.normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1], \
                                               self.Vtx[2:self.Size[0],1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1,1:self.Size[1]-1])
        nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
        # normalized
        norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
        norm_mat_nmle = General.in_mat_zero2one(norm_mat_nmle)
        #norm division
        nmle = General.division_by_norm(nmle,norm_mat_nmle)
        self.Nmls[1:self.Size[0]-1][:,1:self.Size[1]-1] = nmle
        return self.Nmls

    #############################################################################
    ################### Projection and transform Functions #######################
    #############################################################################

    def Draw(self, Pose, s, color = 0) :
        """
        Project vertices and normales in 2D images
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if there is a color image put color in the image
        :return: scene projected in 2D space
        """
        result = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        nmle = np.array([0., 0., 0.])
        for i in range(self.Size[0]/s):
            for j in range(self.Size[1]/s):
                pt[0] = self.Vtx[i*s,j*s][0]
                pt[1] = self.Vtx[i*s,j*s][1]
                pt[2] = self.Vtx[i*s,j*s][2]
                pt = np.dot(Pose, pt)
                pt = pt/pt[:,3].reshape((pt.shape[0], 1))
                nmle[0] = self.Nmls[i*s,j*s][0]
                nmle[1] = self.Nmls[i*s,j*s][1]
                nmle[2] = self.Nmls[i*s,j*s][2]
                nmle = np.dot(Pose[0:3,0:3], nmle)
                if (pt[2] != 0.0):
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(self.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))
                    if (column_index > -1 and column_index < self.Size[1] and line_index > -1 and line_index < self.Size[0]):
                        if (color == 0):
                            result[line_index, column_index] = (self.color_image[i*s,j*s][2], self.color_image[i*s,j*s][1], self.color_image[i*s,j*s][0])
                        else:
                            result[line_index, column_index] = (int((nmle[0] + 1.0)*(255./2.)), int((nmle[1] + 1.0)*(255./2.)), int((nmle[2] + 1.0)*(255./2.)))

        return result


    def Draw_optimize(self, rendering,Pose, s, color = 0) :
        """
        Project vertices and normales from an RGBD image in 2D images
        :param rendering : 2D image for overlay purpose or black image
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if there is a color image put color in the image
        :return: scene projected in 2D space
        """
        result = rendering#np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        stack_pix = np.ones((self.Size[0], self.Size[1]), dtype = np.float32)
        stack_pt = np.ones((np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1)), dtype = np.float32)
        pix = np.zeros((self.Size[0], self.Size[1],2), dtype = np.float32)
        pix = np.dstack((pix,stack_pix))
        pt = np.dstack((self.Vtx[ ::s, ::s, :],stack_pt))
        pt = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        nmle = np.zeros((self.Size[0], self.Size[1],self.Size[2]), dtype = np.float32)
        nmle[ ::s, ::s,:] = np.dot(Pose[0:3,0:3],self.Nmls[ ::s, ::s,:].transpose(0,2,1)).transpose(1,2,0)
        #if (pt[2] != 0.0):
        lpt = np.dsplit(pt,4)
        lpt[2] = General.in_mat_zero2one(lpt[2])
        # if in 1D pix[0] = pt[0]/pt[2]
        pix[ ::s, ::s,0] = (lpt[0]/lpt[2]).reshape(np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1))
        # if in 1D pix[1] = pt[1]/pt[2]
        pix[ ::s, ::s,1] = (lpt[1]/lpt[2]).reshape(np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1))
        pix = np.dot(self.intrinsic,pix[0:self.Size[0],0:self.Size[1]].transpose(0,2,1)).transpose(1,2,0)
        column_index = (np.round(pix[::s,::s,0])).astype(int)
        line_index = (np.round(pix[::s,::s,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < self.Size[1])
        cdt_line = (line_index > -1) * (line_index < self.Size[0])
        cdt = cdt_column*cdt_line
        line_index = line_index*cdt
        column_index = column_index*cdt
        if (color == 0):
            result[line_index[:][:], column_index[:][:]]= np.dstack((self.color_image[ ::s, ::s,2], \
                                                                     self.color_image[ ::s, ::s,1]*cdt_line, \
                                                                     self.color_image[ ::s, ::s,0]*cdt_column) )
        else:
            result[line_index[:][:], column_index[:][:]]= np.dstack( ( (nmle[ ::s, ::s,0]+1.0)*(255./2.)*cdt, \
                                                                       ((nmle[ ::s, ::s,1]+1.0)*(255./2.))*cdt, \
                                                                       ((nmle[ ::s, ::s,2]+1.0)*(255./2.))*cdt ) ).astype(int)
        return result


    def DrawMesh(self, rendering,Vtx,Nmls,Pose, s, color = 2) :
        """
        Project vertices and normales from a mesh in 2D images
        :param rendering : 2D image for overlay purpose or black image
        :param Pose: camera pose
        :param s: subsampling the cloud of points
        :param color: if color=0, put color in the image, if color=1, put boolean in the image
        :return: scene projected in 2D space
        """
        result = rendering#np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)#
        stack_pix = np.ones( (np.size(Vtx[ ::s,:],0)) , dtype = np.float32)
        stack_pt = np.ones( (np.size(Vtx[ ::s,:],0)) , dtype = np.float32)
        pix = np.zeros( (np.size(Vtx[ ::s,:],0),2) , dtype = np.float32)
        pix = np.stack((pix[:,0],pix[:,1],stack_pix),axis = 1)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        pt = np.dot(pt,Pose.T)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Pose[0:3,0:3].T)


        # projection in 2D space
        lpt = np.split(pt,4,axis=1)
        lpt[2] = General.in_mat_zero2one(lpt[2])
        pix[ :,0] = (lpt[0]/lpt[2]).reshape(np.size(Vtx[ ::s,:],0))
        pix[ :,1] = (lpt[1]/lpt[2]).reshape(np.size(Vtx[ ::s,:],0))
        pix = np.dot(pix,self.intrinsic.T)

        column_index = (np.round(pix[:,0])).astype(int)
        line_index = (np.round(pix[:,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < self.Size[1])
        cdt_line = (line_index > -1) * (line_index < self.Size[0])
        cdt = cdt_column*cdt_line
        line_index = line_index*cdt
        column_index = column_index*cdt
        if (color == 0):
            result[line_index[:], column_index[:]]= np.dstack((self.color_image[ line_index[:], column_index[:],2]*cdt, \
                                                                    self.color_image[ line_index[:], column_index[:],1]*cdt, \
                                                                    self.color_image[ line_index[:], column_index[:],0]*cdt) )
        elif (color == 1):
            result[line_index[:], column_index[:]]= 1.0
        else:
            result[line_index[:], column_index[:]]= np.dstack( ( (nmle[ ::s,0]+1.0)*(255./2.)*cdt, \
                                                                       ((nmle[ ::s,1]+1.0)*(255./2.))*cdt, \
                                                                       ((nmle[ ::s,2]+1.0)*(255./2.))*cdt ) ).astype(int)
        return result


    # useless
    def Transform(self, Pose):
        """
        Transform Vertices and Normales with the Pose matrix (generally camera pose matrix)
        :param Pose: 4*4 Transformation Matrix
        :return: none
        """
        stack_pt = np.ones((np.size(self.Vtx,0), np.size(self.Vtx,1)), dtype = np.float32)
        pt = np.dstack((self.Vtx, stack_pt))
        pt = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        self.Vtx = pt[:,:,0:3]
        self.Nmls = np.dot(Pose[0:3,0:3],self.Nmls.transpose(0,2,1)).transpose(1,2,0)

    def project3D22D(self, vtx, Tr= np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)):
        """
        Project vertices from 3d coordinate into 2D images
        :param vtx : points in 3D coordinate
        :param Tr: transformation
        :return: points in 2D coordinate and mask
        """
        size = vtx.shape
        # find the correspondence 2D point by projection
        stack_pix = np.ones(size[0], dtype = np.float32)
        stack_pt = np.ones(size[0], dtype = np.float32)
        pix = np.zeros((size[0], 2), dtype = np.float32)
        pix = np.stack((pix[:,0],pix[:,1],stack_pix), axis = 1)
        pt = np.stack((vtx[:,0],vtx[:,1],vtx[:,2],stack_pt),axis = 1)
        # transform vertices to camera pose
        pt = np.dot(Tr, pt.T).T
        # project to 2D coordinate
        lpt = np.split(pt, 4, axis=1)
        lpt[2] = General.in_mat_zero2one(lpt[2])
        # pix[0] = pt[0]/pt[2]
        pix[:,0] = (lpt[0]/lpt[2]).reshape(size[0])
        pix[:,1] = (lpt[1]/lpt[2]).reshape(size[0])
        pix = np.dot(self.intrinsic,pix.T).T.astype(np.int16)
        mask = (pix[:,0]>=0) * (pix[:,0]<self.Size[1]) * (pix[:,1]>=0) * (pix[:,1]<self.Size[0])

        return pix, mask

##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        """
        Bilateral filtering the depth image
        see cv2 documentation
        """
        self.depth_image = (self.depth_image[:,:] > 0.0) * cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)


##################################################################
################### Segmentation Function #######################
##################################################################
    def RemoveBG(self,binaryImage):
        """
        Delete all the little group (connected component) unwanted from the binary image
        :param binaryImage: a binary image containing several connected component
        :return: A binary image containing only big connected component
        """
        labeled, n = spm.label(binaryImage)
        size = np.bincount(labeled.ravel())
        #do not consider the background
        size2 = np.delete(size,0)
        threshold = max(size2)-1
        keep_labels = size >= threshold
        # Make sure the background is left as 0/False
        keep_labels[0] = 0
        filtered_labeled = keep_labels[labeled]
        return filtered_labeled

    def Crop2Body(self):
        """
        Generate a cropped depthframe from the previous one. The new frame focuses on the human body
        :return: none
        """
        pos2D = self.pos2d[0,self.Index].astype(np.int16)
        # extremes points of the bodies
        minV = np.min(pos2D[:,1])
        maxV = np.max(pos2D[:,1])
        minH = np.min(pos2D[:,0])
        maxH = np.max(pos2D[:,0])
        # distance head to neck. Let us assume this is enough for all borders
        distH2N = LA.norm( (pos2D[self.connection[0,1]-1]-pos2D[self.connection[0,0]-1])).astype(np.int16)+15
        # for MIT data
        '''
        [row, col] = np.where(self.depth_image>0)
        minV = np.min(row)
        maxV = np.max(row)
        minH = np.min(col)
        maxH = np.max(col)
        distH2N = 0
        '''

        Box = self.depth_image
        Box_ori = self.depth_image_ori
        ############ Should check whether the value are in the frame #####################
        colStart = (minH-distH2N).astype(np.int16)
        lineStart = (minV-distH2N).astype(np.int16)
        colEnd = (maxH+distH2N).astype(np.int16)
        lineEnd = (maxV+distH2N).astype(np.int16)
        colStart = max(0, colStart)
        lineStart = max(0, lineStart)
        colEnd = min(colEnd, self.Size[1])
        lineEnd = min(lineEnd, self.Size[0])

        self.transCrop = np.array([colStart,lineStart,colEnd,lineEnd])
        self.CroppedBox = Box[lineStart:lineEnd,colStart:colEnd]
        self.CroppedBox_ori = Box_ori[lineStart:lineEnd,colStart:colEnd]
        if self.hasColor:
            self.CroppedBox_color = self.color_image[lineStart:lineEnd,colStart:colEnd]
        self.CroppedPos = (pos2D -self.transCrop[0:2]).astype(np.int16)

    def BdyThresh(self):
        """
        Threshold the depth image in order to to get the whole body alone with the bounding box (BB)
        :return: The connected component that contain the body
        """
        #'''
        pos2D = self.CroppedPos
        max_value = 1
        self.CroppedBox = self.CroppedBox.astype(np.uint16)
        # Threshold according to detph of the body
        bdyVals = self.CroppedBox[pos2D[self.connection[:,0]-1,1]-1,pos2D[self.connection[:,0]-1,0]-1]
        #only keep vales different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini =  np.min(bdy)
        #print "mini: %u" % (mini)
        maxi = np.max(bdy)
        #print "max: %u" % (maxi)
        # double threshold according to the value of the depth
        bwmin = (self.CroppedBox > mini-0.01*max_value)
        bwmax = (self.CroppedBox < maxi+0.01*max_value)
        bw0 = bwmin*bwmax
        # Remove all stand alone object
        bw0 = ( self.RemoveBG(bw0)>0)
        '''
        #for MIT
        bw0 = (self.CroppedBox>0)
        #'''
        return bw0

    def BodySegmentation(self):
        """
        Calls the function in segmentation.py to process the segmentation of the body
        :return:  none
        """
        #Initialized segmentation with the cropped image
        self.segm = segm.Segmentation(self.CroppedBox,self.CroppedPos)
        # binary image without bqckground
        imageWBG = (self.BdyThresh()>0)

        # Cropped image
        B = self.CroppedBox

        right = 0
        left = 1
        # Process to segmentation algorithm
        armLeft = self.segm.armSeg(imageWBG,B,left)
        armRight = self.segm.armSeg(imageWBG,B,right)
        legRight = self.segm.legSeg(imageWBG,right)
        legLeft = self.segm.legSeg(imageWBG,left)

        # Retrieve every already segmentated part to the main body.
        tmp = armLeft[0]+armLeft[1]+armRight[0]+armRight[1]+legRight[0]+legRight[1]+legLeft[0]+legLeft[1]
        MidBdyImage =(imageWBG-(tmp>0)*1.0)

        # display result
        # cv2.imshow('trunk' , MidBdyImage.astype(np.float))
        # cv2.waitKey(0)

        # continue segmentation for hands and feet
        head = self.segm.headSeg(MidBdyImage)
        handRight = ( self.segm.GetHand( MidBdyImage,right))
        handLeft = ( self.segm.GetHand( MidBdyImage,left))
        footRight = ( self.segm.GetFoot( MidBdyImage,right))
        footLeft = ( self.segm.GetFoot( MidBdyImage,left))

        # handle the ground near the foot
        #''' for MIT
        if self.hasColor:
            a = (footRight*1.0).reshape((self.CroppedBox.shape[0],self.CroppedBox.shape[1],1)) *self.CroppedBox_color
            #cv2.imshow("a", a)
            a = a.reshape((self.CroppedBox.shape[0]*self.CroppedBox.shape[1],3))
            labeled = KMeans(n_clusters=3).fit(a).labels_
            labeled = labeled.reshape((self.CroppedBox.shape[0],self.CroppedBox.shape[1]))
            footRight = (labeled==labeled[self.CroppedPos[19][1]-1, self.CroppedPos[19][0]-1+5])
            cv2.imshow("", labeled*1.0/3)
            cv2.waitKey()
            a = (footLeft*1.0).reshape((self.CroppedBox.shape[0],self.CroppedBox.shape[1],1)) *self.CroppedBox_color
            a = a.reshape((self.CroppedBox.shape[0]*self.CroppedBox.shape[1],3))
            labeled = KMeans(n_clusters=3).fit(a).labels_
            labeled = labeled.reshape((self.CroppedBox.shape[0],self.CroppedBox.shape[1]))
            footLeft = (labeled==labeled[self.CroppedPos[15][1]-1, self.CroppedPos[15][0]-1+5])
        else:
            a = (footRight*1.0) *self.CroppedBox_ori
            a = a.reshape((self.CroppedBox.shape[0]*self.CroppedBox.shape[1],1))
            labeled = KMeans(n_clusters=3).fit(a).labels_
            labeled = labeled.reshape((self.CroppedBox.shape[0],self.CroppedBox.shape[1]))
            footRight = (labeled==labeled[self.CroppedPos[19][1]-1, self.CroppedPos[19][0]-1])
            a = (footLeft*1.0) *self.CroppedBox_ori
            a = a.reshape((self.CroppedBox.shape[0]*self.CroppedBox.shape[1],1))
            labeled = KMeans(n_clusters=3).fit(a).labels_
            labeled = labeled.reshape((self.CroppedBox.shape[0],self.CroppedBox.shape[1]))
            footLeft = (labeled==labeled[self.CroppedPos[15][1]-1, self.CroppedPos[15][0]-1])
        #'''

        # display the trunck
        # cv2.imshow('trunk' , MidBdyImage.astype(np.float))
        # cv2.waitKey(0)

        # Retrieve again every newly computed segmentated part to the main body.
        tmp2 = handRight+handLeft+footRight+footLeft+head
        MidBdyImage2 =(MidBdyImage-(tmp2))

        # Display result
        # cv2.imshow('MidBdyImage2' , MidBdyImage2.astype(np.float))
        # cv2.waitKey(0)
        body = ( self.segm.GetBody( MidBdyImage2)>0)

        # cv2.imshow('body' , body.astype(np.float))
        # cv2.waitKey(0)
        #pdb.set_trace()

        # list of each body parts
        self.bdyPart = np.array( [ armLeft[0], armLeft[1], armRight[0], armRight[1], \
                                   legRight[0], legRight[1], legLeft[0], legLeft[1], \
                                   head, body, handRight, handLeft, footLeft,footRight ]).astype(np.int)#]).astype(np.int)#]).astype(np.int)#
        # list of color for each body parts
        self.bdyColor = np.array( [np.array([0,0,255]), np.array([200,200,255]), np.array([0,255,0]), np.array([200,255,200]),\
                                   np.array([255,0,255]), np.array([255,180,255]), np.array([255,255,0]), np.array([255,255,180]),\
                                   np.array([255,0,0]), np.array([255,255,255]),np.array([0,100,0]),np.array([0,191,255]),\
                                   np.array([255,165,0]),np.array([199,21,133]) ])
        self.labelColor = np.array( ["#0000ff", "#ffc8ff", "#00ff00","#c8ffc8","#ff00ff","#ffb4ff",\
                                   "#ffff00","#ffffb4","#ff0000","#ffffff","#00bfff","#006400",\
                                   "#c715ff","#ffa500"])

        '''
        correspondance between number and body parts and color
        background should have :   color = [0,0,0]       = #000000     black                 label = 0
        armLeft[0] = forearmL      color = [0,0,255]     = #0000ff     blue                  label = 1
        armLeft[1] = upperarmL     color = [200,200,255] = #ffc8ff     very light blue       label = 2
        armRight[0]= forearmR      color = [0,255,0]     = #00ff00     green                 label = 3
        armRight[1] = upperarmR    color = [200,255,200] = #c8ffc8     very light green      label = 4
        legRight[0] = thighR       color = [255,0,255]   = #ff00ff     purple                label = 5
        legRight[1] = calfR        color = [255,180,255] = #ffb4ff     pink                  label = 6
        legLeft[0] = thighL        color = [255,255,0]   = #ffff00     yellow                label = 7
        legLeft[1] = calfL         color = [255,255,180] = #ffffb4     very light yellow     label = 8
        head = headB               color = [255,0,0]     = #ff0000     red                   label = 9
        body = body                color = [255,255,255] = #ffffff     white                 label = 10
        handRight = right hand     color = [0,191,255]   = #00bfff     turquoise             label = 11
        handLeft = left hand       color = [0,100,0]     = #006400     dark green            label = 12
        footRight = right foot     color = [199,21,133]  = #c715ff     dark purple           label = 13
        footLeft = left foot       color = [255,165,0]   = #ffa500     orange                label = 14
        '''


    def BodyLabelling(self):
        '''Create label for each body part in the depth_image'''
        Size = self.depth_image.shape
        self.labels = np.zeros(Size,np.int)
        self.labelList = np.zeros((self.bdyPart.shape[0]+1, Size[0], Size[1]),np.int)
        Txy = self.transCrop
        for i in range(self.bdyPart.shape[0]):
            self.labels[Txy[1]:Txy[3],Txy[0]:Txy[2]] += (i+1)*self.bdyPart[i]
            self.labelList[i+1, Txy[1]:Txy[3],Txy[0]:Txy[2]] += (self.bdyPart[i] + self.overlapmap[i])
            self.labelList[i+1] = (self.labelList[i+1]>0)
            # if some parts overlay, the number of this part will bigger
            overlap = np.where(self.labels > (i+1) )
            #put the overlapping part in the following body part
            self.labels[overlap] = i+1

    def AddOverlap(self):
        """
        add overlap region to each body part
        """
        interLineList = copy.deepcopy([\
        [[self.segm.foreArmPtsL[0], self.segm.foreArmPtsL[1]], [self.segm.foreArmPtsL[2], self.segm.foreArmPtsL[3]]], \
        [[self.segm.upperArmPtsL[0], self.segm.upperArmPtsL[3]], [self.segm.upperArmPtsL[1], self.segm.upperArmPtsL[2]]], \
        [[self.segm.foreArmPtsR[0], self.segm.foreArmPtsR[1]], [self.segm.foreArmPtsR[2], self.segm.foreArmPtsR[3]]], \
        [[self.segm.upperArmPtsR[0], self.segm.upperArmPtsR[3]], [self.segm.upperArmPtsR[2], self.segm.upperArmPtsR[1]]], \
        [[self.segm.thighPtsR[0], self.segm.thighPtsR[1]], [self.segm.thighPtsR[2], self.segm.thighPtsR[3]]], \
        [[self.segm.calfPtsR[0], self.segm.calfPtsR[1]], [self.segm.calfPtsR[2], self.segm.calfPtsR[3]]],
        [[self.segm.thighPtsL[0],  self.segm.thighPtsL[1]], [self.segm.thighPtsL[2],self.segm.thighPtsL[3]]], \
        [[self.segm.calfPtsL[0], self.segm.calfPtsL[1]], [self.segm.calfPtsL[2], self.segm.calfPtsL[3]]], \
        [[self.segm.peakshoulderL.copy(), self.segm.peakshoulderR.copy()]], \
        [[self.segm.upperArmPtsL[2], self.segm.upperArmPtsL[1]], [self.segm.peakshoulderL.copy(), self.segm.peakshoulderR.copy()], [self.segm.upperArmPtsR[1], self.segm.upperArmPtsR[2]], [self.segm.thighPtsR[1], self.segm.thighPtsR[0]], [self.segm.thighPtsR[0], self.segm.thighPtsL[1]]], \
        [[self.segm.foreArmPtsR[3], self.segm.foreArmPtsR[2]]], \
        [[self.segm.foreArmPtsL[3], self.segm.foreArmPtsL[2]]], \
        [[self.segm.calfPtsL[1], self.segm.calfPtsL[0]]], \
        [[self.segm.calfPtsR[1], self.segm.calfPtsR[0]]], \
        ])
        self.overlapmap = np.zeros((14, self.CroppedBox.shape[0], self.CroppedBox.shape[1]), np.int)
        a = np.zeros((self.CroppedBox.shape[0], self.CroppedBox.shape[1]), np.int)
        for i in range(len(interLineList)):
            interLines = interLineList[i]
            for j in range(len(interLines)):
                interPoints = interLines[j]
                rr,cc,val = line_aa(int(interPoints[0][1]), int(interPoints[0][0]), int(interPoints[1][1]), int(interPoints[1][0]))
                self.overlapmap[i, rr, cc]=2
            '''
            Txy = self.transCrop
            a += self.overlapmap[i]
            a += self.bdyPart[i]
        cv2.imshow("", a.astype(np.double)/2)
        cv2.waitKey()
        '''

    def RGBDSegmentation(self):
        """
        Call every method to have a complete segmentation
        :return: none
        """
        self.Crop2Body()
        self.BodySegmentation()
        self.AddOverlap()
        self.BodyLabelling()


#######################################################################
################### Bounding boxes Function #######################
##################################################################

    def GetCenter3D(self,i):
        """
        Compute the mean for one segmented part
        :param i: number of the body part
        :return: none
        """
        ctr_x = (max(self.PtCloud[i][:, 0])+min(self.PtCloud[i][:, 0]))/2
        ctr_y = (max(self.PtCloud[i][:, 1])+min(self.PtCloud[i][:, 1]))/2
        ctr_z = (max(self.PtCloud[i][:, 2])+min(self.PtCloud[i][:, 2]))/2
        return [ctr_x, ctr_y, ctr_z]


    def SetTransfoMat3D(self,evecs,i):
        """
        Generate the transformation matrix
        :param evecs: eigen vectors
        :param i: number of the body part
        :return: none
        """
        ctr = self.ctr3D[i]#self.coordsGbl[i][0]#[0.,0.,0.]#
        e1 = evecs[0]
        e2 = evecs[1]
        e3 = evecs[2]
        # axis of coordinates system
        e1b = np.array( [e1[0],e1[1],e1[2],0])
        e2b = np.array( [e2[0],e2[1],e2[2],0])
        e3b = np.array( [e3[0],e3[1],e3[2],0])
        #center of coordinates system
        origine = np.array( [ctr[0],ctr[1],ctr[2],1])
        # concatenate it in the right order.
        Transfo = np.stack( (e1b,e2b,e3b,origine),axis = 0 )
        self.TransfoBB.append(Transfo.transpose())
        #display
        #print "TransfoBB[%d]" %(i)
        #print self.TransfoBB[i]


    def bdyPts3D(self, mask):
        """
        create of cloud of point from part of the RGBD image
        :param mask: a matrix containing one only in the body parts indexes, 0 otherwise
        :return:  list of vertices = cloud of points
        """
        start_time2 = time.time()
        nbPts = sum(sum(mask))
        res = np.zeros((nbPts, 3), dtype = np.float32)
        k = 0
        for i in range(self.Size[0]):
            for j in range(self.Size[1]):
                if(mask[i,j]):
                    res[k] = self.Vtx[i,j]
                    k = k+1
        elapsed_time3 = time.time() - start_time2
        print "making pointcloud process time: %f" % (elapsed_time3)
        return res

    def bdyPts3D_optimize(self, mask):
        """
        create of cloud of point from part of the RGBD image
        :param mask: a matrix containing one only in the body parts indexes, 0 otherwise
        :return:  list of vertices = cloud of points
        """
        #start_time2 = time.time()
        nbPts = sum(sum(mask))

        # threshold with the mask
        x = self.Vtx[:,:,0]*mask
        y = self.Vtx[:,:,1]*mask
        z = self.Vtx[:,:,2]*mask

        #keep only value that are different from 0 in the list
        x_res = x[~(z==0)]
        y_res = y[~(z==0)]
        z_res = z[~(z==0)]

        #concatenate each axis
        res = np.dstack((x_res,y_res,z_res)).reshape(nbPts,3)

        #elapsed_time3 = time.time() - start_time2
        #print "making pointcloud process time: %f" % (elapsed_time3)

        return res

    def getSkeletonVtx(self):
        """
        calculate the skeleton in 3D
        :retrun: none
        """
        # get pos2D
        pos2D = self.pos2d[0,self.Index].astype(np.double)-1
        # initialize
        skedepth = np.zeros(25)

        # compute depth of each junction
        for i in range(21): # since 21~24 uesless
            if i==0 or i == 1 or i == 20:
                j=10
            elif i==2 or i==3:
                j=9
            elif i==4 or i==5:
                j=2
            elif i==6:
                j=1
            elif i==7 or i==21 or i==22:
                j=12
            elif i==8 or i==9:
                j=4
            elif i==10:
                j=3
            elif i==11 or i==23 or i==24:
                j=11
            elif i==12:
                j=7
            elif i==13 or i==14:
                j=8
            elif i==15:
                j=13
            elif i==16:
                j=5
            elif i==17 or i==18:
                j=6
            elif i==19:
                j=14

            depth = abs(np.amax(self.coordsGbl[j][:,2])-np.amin(self.coordsGbl[j][0,2]))/2
            depth = 0
            if self.labels[int(pos2D[i][1]), int(pos2D[i][0])]!=0:
                skedepth[i] = self.depth_image[int(pos2D[i][1]), int(pos2D[i][0])]+depth
            else:
                print "meet the pose " + str(i) + "==0 when getting junction"
                if self.labels[int(pos2D[i][1])+1, int(pos2D[i][0])]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1])+1, int(pos2D[i][0])]+depth
                elif self.labels[int(pos2D[i][1]), int(pos2D[i][0])+1]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1]), int(pos2D[i][0])+1]+depth
                elif self.labels[int(pos2D[i][1])-1, int(pos2D[i][0])]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1])-1, int(pos2D[i][0])]+depth
                elif self.labels[int(pos2D[i][1]), int(pos2D[i][0])-1]!=0:
                    skedepth[i] = self.depth_image[int(pos2D[i][1]), int(pos2D[i][0])-1]+depth
                else:
                    print "QAQQQQ"
                    #exit()

        #  project to 3D
        pos2D[:,0] = (pos2D[:,0]-self.intrinsic[0,2])/self.intrinsic[0,0]
        pos2D[:,1] = (pos2D[:,1]-self.intrinsic[1,2])/self.intrinsic[1,1]
        x = skedepth * pos2D[:,0]
        y = skedepth * pos2D[:,1]
        z = skedepth

        # give hand and foot't joint correct
        for i in [7,11,15,19]:
            x[i] = (x[i-1]-x[i-2])/4+x[i-1]
            y[i] = (y[i-1]-y[i-2])/4+y[i-1]
            z[i] = (z[i-1]-z[i-2])/4+z[i-1]

        return np.dstack((x,y,z)).astype(np.float32)


    def myPCA(self, dims_rescaled_data=3):
        # dims_rescaled_data useless
        """
        Compute the principal component analysis on a cloud of points
        to get the coordinates system local to the cloud of points
        :param dims_rescaled_data: 3 per default, number of dimension wanted
        :return:  none
        """
        # list of center in the 3D space
        self.ctr3D = []
        self.ctr3D.append([0.,0.,0.])
        # list of transformed Vtx of each bounding boxes
        self.TVtxBB = []
        self.TVtxBB.append([0.,0.,0.])
        # list of coordinates sys with center
        self.TransfoBB = []
        self.TransfoBB.append([0.,0.,0.])
        self.vects3D = []
        self.vects3D.append([0.,0.,0.])
        self.PtCloud = []
        self.PtCloud.append([0.,0.,0.])
        self.pca = []
        self.pca.append(PCA(n_components=3))
        self.coordsL=[]
        self.coordsL.append([0.,0.,0.])
        self.coordsGbl=[]
        self.coordsGbl.append([0.,0.,0.])
        self.mask=[]
        self.mask.append([0.,0.,0.])
        self.BBsize = []
        self.BBsize.append([0.,0.,0.])
        for i in range(1,self.bdyPart.shape[0]+1):
            self.mask.append( (self.labels == i) )
            # compute center of 3D
            self.PtCloud.append(self.bdyPts3D_optimize(self.mask[i]))
            self.pca.append(PCA(n_components=3))
            self.pca[i].fit(self.PtCloud[i])

            # Compute 3D centers
            #self.ctr3D.append(self.GetCenter3D(i))
            self.ctr3D.append(self.pca[i].mean_)
            #print "ctr3D indexes :"
            #print self.ctr3D[i]

            # eigen vectors
            self.vects3D.append(self.pca[i].components_)
            #global to local transform of the cloud of point
            self.TVtxBB.append( self.pca[i].transform(self.PtCloud[i]))

            #Coordinates of the bounding boxes
            self.FindCoord3D(i)
            #Create local to global transform
            self.SetTransfoMat3D(self.pca[i].components_,i)

        # create the skeleton vtx
        self.skeVtx = self.getSkeletonVtx()

    def FindCoord3D(self,i):
        '''
        draw the bounding boxes in 3D for each part of the human body
        :param i : number of the body parts
        '''
        # Adding a space so that the bounding boxes are wider
        VoxSize = 0.005
        wider = 5*VoxSize*0
        # extremes planes of the bodies
        minX = np.min(self.TVtxBB[i][:,0]) - wider
        maxX = np.max(self.TVtxBB[i][:,0]) + wider
        minY = np.min(self.TVtxBB[i][:,1]) - wider
        maxY = np.max(self.TVtxBB[i][:,1]) + wider
        minZ = np.min(self.TVtxBB[i][:,2]) - wider
        maxZ = np.max(self.TVtxBB[i][:,2]) + wider
        # extremes points of the bodies
        xymz = np.array([minX,minY,minZ])
        xYmz = np.array([minX,maxY,minZ])
        Xymz = np.array([maxX,minY,minZ])
        XYmz = np.array([maxX,maxY,minZ])
        xymZ = np.array([minX,minY,maxZ])
        xYmZ = np.array([minX,maxY,maxZ])
        XymZ = np.array([maxX,minY,maxZ])
        XYmZ = np.array([maxX,maxY,maxZ])

        # New coordinates and new images
        self.coordsL.append( np.array([xymz,xYmz,XYmz,Xymz,xymZ,xYmZ,XYmZ,XymZ]) )
        #print "coordsL[%d]" %(i)
        #print self.coordsL[i]

        # transform back
        self.coordsGbl.append( self.pca[i].inverse_transform(self.coordsL[i]))
        #print "coordsGbl[%d]" %(i)
        #print self.coordsGbl[i]

        # save the boundingboxes size
        self.BBsize.append([LA.norm(self.coordsGbl[i][3] - self.coordsGbl[i][0]), LA.norm(self.coordsGbl[i][1] - self.coordsGbl[i][0]), LA.norm(self.coordsGbl[i][4] - self.coordsGbl[i][0])])

    def BuildBB(self):
        """
        build bounding boxes to let no overlapping bounding boxes
        :return: none
        """
        # settings
        interPointList = copy.deepcopy([[], \
        [self.segm.foreArmPtsL[0], self.segm.foreArmPtsL[1], self.segm.foreArmPtsL[2], self.segm.foreArmPtsL[3]], \
        [self.segm.upperArmPtsL[0], self.segm.upperArmPtsL[1], self.segm.upperArmPtsL[2], self.segm.upperArmPtsL[3]], \
        [self.segm.foreArmPtsR[0], self.segm.foreArmPtsR[1], self.segm.foreArmPtsR[2], self.segm.foreArmPtsR[3]], \
        [self.segm.upperArmPtsR[0], self.segm.upperArmPtsR[3], self.segm.upperArmPtsR[2], self.segm.upperArmPtsR[1]], \
        [self.segm.thighPtsR[0], self.segm.thighPtsR[1], self.segm.thighPtsR[2], self.segm.thighPtsR[3]], \
        [self.segm.calfPtsR[0], self.segm.calfPtsR[1], self.segm.calfPtsR[2], self.segm.calfPtsR[3]],
        [self.segm.thighPtsL[0], self.segm.thighPtsL[3], self.segm.thighPtsL[2], self.segm.thighPtsL[1]], \
        [self.segm.calfPtsL[0], self.segm.calfPtsL[1], self.segm.calfPtsL[2], self.segm.calfPtsL[3]], \
        [self.segm.peakshoulderL.copy(), self.segm.headPts[1], self.segm.headPts[0], self.segm.peakshoulderR.copy()], \
        [self.segm.upperArmPtsL[2], self.segm.upperArmPtsL[1], self.segm.peakshoulderL.copy(), self.segm.peakshoulderR.copy(), self.segm.upperArmPtsR[1], self.segm.upperArmPtsR[2], self.segm.thighPtsR[1], self.segm.thighPtsR[0], self.segm.thighPtsL[1]], \
        [self.segm.foreArmPtsR[3], self.segm.foreArmPtsR[2]], \
        [self.segm.foreArmPtsL[3], self.segm.foreArmPtsL[2]], \
        [self.segm.calfPtsL[1], self.segm.calfPtsL[0]], \
        [self.segm.calfPtsR[1], self.segm.calfPtsR[0]], \
        ])
        interPointList2D = copy.deepcopy(interPointList)
        labelList = [[],[2,2,12,12], [2,2,2,2], [4,4,11,11], [4,4,4,4], [5,5,5,5], [6,6,5,5], [5,7,7,7], [8,8,7,7], \
        [9,9,9,9], [2,2,9,9,4,4,5,5,7], [11,11,11,11], [12,12,12,12], [8,8,13,13], [6,6,14,14]]
        t=0


        #2D 2 3D
        for i in range(1, len(interPointList)):
            for j in range(len(interPointList[i])):
                depth = sum(sum(self.depth_image*(self.labels==labelList[i][j])))/sum(sum(self.labels==labelList[i][j]))

                # point = [x,y]
                # move positions from cropped box to original size
                interPointList[i][j] = map(float, interPointList[i][j])
                interPointList[i][j][0] = float(interPointList[i][j][0] + self.transCrop[0])
                interPointList[i][j][1] = float(interPointList[i][j][1] + self.transCrop[1])
                # project to 3D coordinate
                interPointList[i][j][0] = ( interPointList[i][j][0] - self.intrinsic[0,2])/self.intrinsic[0,0]*depth
                interPointList[i][j][1] = ( interPointList[i][j][1] - self.intrinsic[1,2])/self.intrinsic[1,1]*depth
                interPointList[i][j].append(depth)
                t+=1

        # for each body part
        self.coordsGbl = []
        self.coordsGbl.append(np.array((0,0,0)))
        self.BBTrans = []
        self.BBTrans.append(np.identity(4))

        for bp in range(1,len(interPointList)):
            points = interPointList[bp]

            if bp==11 or bp==12:
                if bp==12:
                    point2 = [interPointList[1][0][0]/2+interPointList[1][1][0]/2, interPointList[1][0][1]/2+interPointList[1][1][1]/2, interPointList[1][0][2]/2+interPointList[1][1][2]/2]
                if bp==11:
                    point2 = [interPointList[3][0][0]/2+interPointList[3][1][0]/2, interPointList[3][0][1]/2+interPointList[3][1][1]/2, interPointList[3][0][2]/2+interPointList[3][1][2]/2]
                point1 = [points[0][0]/2+points[1][0]/2, points[0][1]/2+points[1][1]/2, points[0][2]/2+points[1][2]/2]
                vector = [point1[0]-point2[0],point1[1]-point2[1],point1[2]-point2[2]]
                vector = vector/np.linalg.norm(vector)*0.25
                points.append(points[1]+vector)
                points.append(points[0]+vector)
                if abs(vector[0])>abs(vector[1]):
                    if points[2][1]<points[3][1]:
                        points[2][1] -= 0.1
                        points[3][1] += 0.1
                    else:
                        points[2][1] += 0.1
                        points[3][1] -= 0.1
                else:
                    if points[2][0]>points[3][0]:
                        points[2][0] += 0.1
                        points[3][0] -= 0.1
                    else:
                        points[2][0] -= 0.1
                        points[3][0] += 0.1

            if bp==13 or bp==14:
                if bp==13:
                    point2 = [interPointList[8][2][0]/2+interPointList[8][3][0]/2, interPointList[8][2][1]/2+interPointList[8][3][1]/2, interPointList[8][2][2]/2+interPointList[8][3][2]/2]
                if bp==14:
                    point2 = [interPointList[6][2][0]/2+interPointList[6][3][0]/2, interPointList[6][2][1]/2+interPointList[6][3][1]/2, interPointList[6][2][2]/2+interPointList[6][3][2]/2]
                point1 = [points[0][0]/2+points[1][0]/2, points[0][1]/2+points[1][1]/2, points[0][2]/2+points[1][2]/2]
                vector = [point1[0]-point2[0],point1[1]-point2[1],point1[2]-point2[2]]
                vector = vector/np.linalg.norm(vector)*0.25
                points.append(points[1]+vector+[0.05,0,0])
                points.append(points[0]+vector+[-0.05,0,0])

            coordGbl =  np.zeros((len(points)*2,3), dtype=np.float32)
            BBTrans = np.zeros((len(points)*2,4,4), dtype=np.float32)
            # for each line of one body part
            for p in range(len(points)):
                # get depth
                if (bp==11 or bp==12 or bp==13 or bp==14) and (p==2 or p==3):
                    point2d = interPointList2D[bp][3-p]
                else:
                    point2d = interPointList2D[bp][p]
                if (bp==6 or bp==8 or bp==13 or bp==14) and (p==0 or p==1):
                    depthMax = np.amax(np.amax(self.depth_image*(self.labels==labelList[bp][p])))
                    depthMin = np.amin(np.amin(self.depth_image[np.nonzero(self.depth_image*(self.labels==labelList[bp][p]))]))
                elif bp==9 and (p==1 or p==2):
                    depthMax = np.amax(np.amax(self.depth_image*(self.labels==labelList[bp][p])))
                    depthMin = np.amin(np.amin(self.depth_image[np.nonzero(self.depth_image*(self.labels==labelList[bp][p]))]))
                else:
                    line = self.depth_image.shape[0]
                    col = self.depth_image.shape[1]
                    mask = np.ones([line,col,2])
                    mask = mask*point2d
                    mask[:,:,0]+= self.transCrop[0]
                    mask[:,:,1]+= self.transCrop[1]
                    lineIdx = np.array([np.arange(line) for _ in range(col)]).transpose()
                    colIdx = np.array([np.arange(col) for _ in range(line)])
                    ind = np.stack( (colIdx,lineIdx), axis = 2)
                    mask = np.sqrt(np.sum( (ind-mask)*(ind-mask),axis = 2))
                    mask = (mask < 16)
                    depthMax = np.amax(np.amax(self.depth_image*mask))
                    depthMin = np.amin(np.amin(self.depth_image[np.nonzero(self.depth_image*mask)]))

                point = points[p]
                coordGbl[p] = np.array([point[0], point[1], depthMin])
                coordGbl[p+len(points)] = np.array([point[0], point[1], depthMax])
                BBTrans[p] = np.identity(4)
                BBTrans[p+len(points)] = np.identity(4)
            self.coordsGbl.append(coordGbl)
            self.BBTrans.append(BBTrans)

        # update local coordinate
        self.coordsL = []
        self.coordsL.append([0.,0.,0.])
        self.BBsize = []
        self.BBsize.append([0.,0.,0.])
        for bp in range(1,15):
            self.coordsL.append(self.pca[bp].transform(self.coordsGbl[bp]).astype(np.float32))
            minX = np.min(self.coordsL[bp][:,0])
            maxX = np.max(self.coordsL[bp][:,0])
            minY = np.min(self.coordsL[bp][:,1])
            maxY = np.max(self.coordsL[bp][:,1])
            minZ = np.min(self.coordsL[bp][:,2])
            maxZ = np.max(self.coordsL[bp][:,2])
            self.BBsize.append([LA.norm(maxX - minX), LA.norm(maxY - minY), LA.norm(maxZ - minZ)])


    def getWarpingPlanes(self):
        """
        Get the area function which is used to compute weight when warping, in all body part
        :param self.skeVtx self.coordsGbl
        :retrun self.planesF
        """
        self.planesF = np.zeros((15,4), dtype=np.float32)
        for bp in range(1,15):
            if bp==1:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 1
                planeIdx[0,1] = 0
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][5]-self.skeVtx[0][4]
                boneV = self.skeVtx[0][6]-self.skeVtx[0][5]
                point = self.skeVtx[0][5]
            elif bp==2:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 2
                planeIdx[0,1] = 1
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 0
                boneV_p = self.skeVtx[0][20]-self.skeVtx[0][1]
                boneV_p[0], boneV_p[1] = boneV_p[1], boneV_p[0]
                boneV_p[2] = 0
                boneV = self.skeVtx[0][5]-self.skeVtx[0][4]
                point = self.skeVtx[0][4]
            elif bp==3:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 0
                planeIdx[0,1] = 1
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][9]-self.skeVtx[0][8]
                boneV = self.skeVtx[0][10]-self.skeVtx[0][9]
                point = self.skeVtx[0][9]
            elif bp==4:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 3
                planeIdx[0,1] = 2
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 0
                boneV_p = self.skeVtx[0][20]-self.skeVtx[0][1]
                boneV_p[0], boneV_p[1] = -boneV_p[1], -boneV_p[0]
                boneV_p[2] = 0
                boneV = self.skeVtx[0][9]-self.skeVtx[0][8]
                point = self.skeVtx[0][8]
            elif bp==5:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 1
                planeIdx[0,1] = 0
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][0]-self.skeVtx[0][1]
                boneV = self.skeVtx[0][17]-self.skeVtx[0][16]
                point = self.skeVtx[0][16]
            elif bp==6:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 3
                planeIdx[0,1] = 2
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 0
                boneV_p = self.skeVtx[0][17]-self.skeVtx[0][16]
                boneV = self.skeVtx[0][18]-self.skeVtx[0][17]
                point = self.skeVtx[0][17]
            elif bp==7:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 0
                planeIdx[0,1] = 3
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][0]-self.skeVtx[0][1]
                boneV = self.skeVtx[0][13]-self.skeVtx[0][12]
                point = self.skeVtx[0][12]
            elif bp==8:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 3
                planeIdx[0,1] = 2
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 0
                boneV_p = self.skeVtx[0][13]-self.skeVtx[0][12]
                boneV = self.skeVtx[0][14]-self.skeVtx[0][13]
                point = self.skeVtx[0][13]
            elif bp==9:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 0
                planeIdx[0,1] = 3
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 1
                boneV_p = self.skeVtx[0][20]-self.skeVtx[0][1]
                boneV = self.skeVtx[0][3]-self.skeVtx[0][2]
                point = self.skeVtx[0][2]
            elif bp==10:
                planeIdx = np.zeros((2,3), dtype = np.float32)
                planeIdx[0,0] = self.skeVtx[0][0,0]
                planeIdx[0,1] = self.skeVtx[0][0,1]
                planeIdx[0,2] = self.skeVtx[0][0,2]
                planeIdx[1,0] = self.skeVtx[0][1,0]
                planeIdx[1,1] = self.skeVtx[0][1,1]
                planeIdx[1,2] = self.skeVtx[0][1,2]
            elif bp==11:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 0
                planeIdx[0,1] = 1
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][9]-self.skeVtx[0][10]
                boneV = self.skeVtx[0][10]-self.skeVtx[0][11]
                point = self.skeVtx[0][10]
            elif bp==12:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 1
                planeIdx[0,1] = 0
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][5]-self.skeVtx[0][6]
                boneV = self.skeVtx[0][6]-self.skeVtx[0][7]
                point = self.skeVtx[0][6]
            elif bp==13:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 1
                planeIdx[0,1] = 0
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][13]-self.skeVtx[0][14]
                boneV = self.skeVtx[0][14]-self.skeVtx[0][15]
                point = self.skeVtx[0][14]
            elif bp==14:
                planeIdx = np.zeros((1,5), dtype = np.float32)
                planeIdx[0,0] = 1
                planeIdx[0,1] = 0
                planeIdx[0, 2:4] = planeIdx[0,0:2]+4
                planeIdx[0,4] = 2
                boneV_p = self.skeVtx[0][17]-self.skeVtx[0][18]
                boneV = self.skeVtx[0][18]-self.skeVtx[0][19]
                point = self.skeVtx[0][18]
            if bp!=10:
                v1 = self.coordsGbl[bp][int(planeIdx[0,1])] - self.coordsGbl[bp][int(planeIdx[0,0])]
                v2 = self.coordsGbl[bp][int(planeIdx[0,2])] - self.coordsGbl[bp][int(planeIdx[0,0])]
                self.planesF[bp,0:3] = np.cross(v1, v2)
                self.planesF[bp,0:3] /= LA.norm(self.planesF[bp,0:3])
                self.planesF[bp, 3] = -np.dot(self.planesF[bp, 0:3], self.coordsGbl[bp][int(planeIdx[0,1])])

                #plane3
                if bp!=5 and bp!=7:
                    self.planesF[bp,0:3] = boneV[0:3]
                    self.planesF[bp,0:3] /= LA.norm(self.planesF[bp,0:3])
                    self.planesF[bp, 3] = -np.dot(self.planesF[bp, 0:3], point)
                else:
                    self.planesF[bp, 3] = -np.dot(self.planesF[bp, 0:3], self.coordsGbl[bp][int(planeIdx[0,1])])


                if np.dot(self.planesF[bp,0:3], self.coordsGbl[bp][int(planeIdx[0,4])])+self.planesF[bp,3] <0:
                    self.planesF[bp] = -self.planesF[bp]

            else:
                self.planesF[bp,0:3] = planeIdx[0,:]-planeIdx[1,:]
                self.planesF[bp,0:3] /= LA.norm(self.planesF[bp,0:3])
                self.planesF[bp, 3] = -np.dot(self.planesF[bp, 0:3], planeIdx[1,:])

    def GetProjPts2D(self, vects3D, Pose, s=1) :
        """
        Project a list of vertexes in the image RGBD
        :param vects3D: list of 3 elements vector
        :param Pose: Transformation matrix
        :param s: subsampling coefficient
        :return: transformed list of 3D vector
        """
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        drawVects = []
        for i in range(len(vects3D)):
            pt[0] = vects3D[i][0]
            pt[1] = vects3D[i][1]
            pt[2] = vects3D[i][2]
            # transform list
            pt = np.dot(Pose, pt)
            pt /= pt[:,3].reshape((pt.shape[0], 1))
            #Project it in the 2D space
            if (pt[2] != 0.0):
                pix[0] = pt[0]/pt[2]
                pix[1] = pt[1]/pt[2]
                pix = np.dot(self.intrinsic, pix)
                column_index = pix[0].astype(np.int)
                line_index = pix[1].astype(np.int)
            else :
                column_index = 0
                line_index = 0
            #print "line,column index : (%d,%d)" %(line_index,column_index)
            drawVects.append(np.array([column_index,line_index]))
        return drawVects

    def GetProjPts2D_optimize(self, vects3D, Pose) :
        """
        Project a list of vertexes in the image RGBD. Optimize for CPU version.
        :param vects3D: list of 3 elements vector
        :param Pose: Transformation matrix
        :return: transformed list of 3D vector
        """
        '''Project a list of vertexes in the image RGBD'''
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        pix = np.stack((pix for i in range(len(vects3D)) ))
        pt = np.stack((pt for i in range(len(vects3D)) ))
        pt[:,0:3] = vects3D
        # transform list
        pt = np.dot(pt,Pose.T)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        # Project it in the 2D space
        pt[:,2] = General.in_mat_zero2one(pt[:,2])
        pix[:,0] = pt[:,0]/pt[:,2]
        pix[:,1] = pt[:,1]/pt[:,2]
        pix = np.dot( pix,self.intrinsic.T)
        column_index = pix[:,0].astype(np.int)
        line_index = pix[:,1].astype(np.int)
        drawVects = np.array([column_index,line_index]).T
        return drawVects



    def GetNewSys(self, Pose,ctr2D,nbPix) :
        '''
        compute the coordinates of the points that will create the coordinates system
        '''
        self.drawNewSys = []
        maxDepth = max(0.0001, np.max(self.Vtx[:,:,2]))

        for i in range(1,len(self.vects3D)):
            vect = np.dot(self.vects3D[i],Pose[0:3,0:3].T )
            vect /= vect[:,3].reshape((vect.shape[0], 1))
            newPt = np.zeros(vect.shape)
            for j in range(vect.shape[0]):
                newPt[j][0] = ctr2D[i][0]-nbPix*vect[j][0]
                newPt[j][1] = ctr2D[i][1]-nbPix*vect[j][1]
                newPt[j][2] = vect[j][2]-nbPix*vect[j][2]/maxDepth
            self.drawNewSys.append(newPt)



    def Cvt2RGBA(self,im_im):
        '''
        convert an RGB image in RGBA to put all zeros as transparent
        THIS FUNCTION IS NOT USED IN THE PROJECT
        '''
        img = im_im.convert("RGBA")
        datas = img.getdata()
        newData = []
        for item in datas:
            if item[0] == 0 and item[1] == 0 and item[2] == 0:
                newData.append((0, 0, 0, 0))
            else:
                newData.append(item)

        img.putdata(newData)
        return img

