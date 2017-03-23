# File created by Diego Thomas the 16-11-2016
# improved by Inoe Andre from 02-2017

# Define functions to manipulate RGB-D data
import cv2
import numpy as np
from numpy import linalg as LA
import random
import imp
import time
import scipy.ndimage.measurements as spm
import pdb

segm = imp.load_source('segmentation', './lib/segmentation.py')

def normalized_cross_prod(a,b):
    res = np.zeros(3, dtype = "float")
    if (LA.norm(a) == 0.0 or LA.norm(b) == 0.0):
        return res
    a = a/LA.norm(a)
    b = b/LA.norm(b)
    res[0] = a[1]*b[2] - a[2]*b[1]
    res[1] = -a[0]*b[2] + a[2]*b[0]
    res[2] = a[0]*b[1] - a[1]*b[0]
    if (LA.norm(res) > 0.0):
        res = res/LA.norm(res)
    return res


def in_mat_zero2one(mat):
    """This fonction replace in the matrix all the 0 to 1"""
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res

def division_by_norm(mat,norm):
    """This fonction divide a n by m by p=3 matrix, point by point, by the norm made through the p dimension>
    It ignores division that makes infinite values or overflow to replace it by the former mat values or by 0"""
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            mat[:,:,i] = np.true_divide(mat[:,:,i],norm)
            mat[:,:,i][mat[:,:,i] == np.inf] = 0
            mat[:,:,i] = np.nan_to_num(mat[:,:,i])
    return mat
                
def normalized_cross_prod_optimize(a,b):
    #res = np.zeros(a.Size, dtype = "float")
    norm_mat_a = np.sqrt(np.sum(a*a,axis=2))
    norm_mat_b = np.sqrt(np.sum(b*b,axis=2))
    #changing every 0 to 1 in the matrix so that the division does not generate nan or infinite values
    norm_mat_a = in_mat_zero2one(norm_mat_a)
    norm_mat_b = in_mat_zero2one(norm_mat_b)
    # compute a/ norm_mat_a
    a = division_by_norm(a,norm_mat_a)
    b = division_by_norm(b,norm_mat_b)
    #compute cross product with matrix
    res = np.cross(a,b)
    #compute the norm of res using the same method for a and b 
    norm_mat_res = np.sqrt(np.sum(res*res,axis=2))
    norm_mat_res = in_mat_zero2one(norm_mat_res)
    #norm division
    res = division_by_norm(res,norm_mat_res)
    return res

#Nurbs class to handle NURBS curves (Non-uniform rational B-spline)
class RGBD():

    # Constructor
    def __init__(self, depthname, colorname, intrinsic, fact):
        self.depthname = depthname
        self.colorname = colorname
        self.intrinsic = intrinsic
        self.fact = fact
        
    def LoadMat(self, Images,Pos_2D,BodyConnection,binImage):
        self.lImages = Images
        self.numbImages = len(self.lImages.transpose())
        self.Index = -1
        self.pos2d = Pos_2D
        self.connection = BodyConnection
        self.bw = binImage
        
    def ReadFromDisk(self): #Read an RGB-D image from the disk
        print(self.depthname)
        self.depth_in = cv2.imread(self.depthname, -1)
        self.color_image = cv2.imread(self.colorname, -1)
        
        self.Size = self.depth_in.shape
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]):
                self.depth_image[i,j] = float(self.depth_in[i,j][0]) / self.fact
                                
    def ReadFromMat(self, idx = -1):
        if (idx == -1):
            self.Index = self.Index + 1
        else:
            self.Index = idx
            
        depth_in = self.lImages[0][self.Index]
        print "Input depth image is of size: ", depth_in.shape
        size_depth = depth_in.shape
        self.Size = (size_depth[0], size_depth[1], 3)
        self.depth_image = np.zeros((self.Size[0], self.Size[1]), np.float32)
        self.depth_image = depth_in.astype(np.float32) / self.fact
        self.skel = self.depth_image.copy()

    def DrawSkeleton(self):
        '''this function draw the Skeleton of a human and make connections between each part'''
        pos = self.pos2d[0][self.Index]
        for i in range(np.size(self.connection,0)):
            pt1 = (pos[self.connection[i,0]-1,0],pos[self.connection[i,0]-1,1])
            pt2 = (pos[self.connection[i,1]-1,0],pos[self.connection[i,1]-1,1])
            cv2.line( self.skel,pt1,pt2,(0,0,255),2) # color space = BGR
            cv2.circle(self.skel,pt1,1,(0,0,255),2)
            cv2.circle(self.skel,pt2,1,(0,0,255),2)


    def Vmap(self): # Create the vertex image from the depth image and intrinsic matrice
        self.Vtx = np.zeros(self.Size, np.float32)
        for i in range(self.Size[0]): # line index (i.e. vertical y axis)
            for j in range(self.Size[1]): # column index (i.e. horizontal x axis)
                d = self.depth_image[i,j]
                if d > 0.0:
                    x = d*(j - self.intrinsic[0,2])/self.intrinsic[0,0]
                    y = d*(i - self.intrinsic[1,2])/self.intrinsic[1,1]
                    self.Vtx[i,j] = (x, y, d)
        
    
    def Vmap_optimize(self): # Create the vertex image from the depth image and intrinsic matrice
        self.Vtx = np.zeros(self.Size, np.float32)
        d = self.skel[0:self.Size[0]][0:self.Size[1]]
        d_pos = d * (d > 0.0)
        x_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        y_raw = np.zeros([self.Size[0],self.Size[1]], np.float32)
        # change the matrix so that the first row is on all rows for x respectively colunm for y.
        x_raw[0:-1,:] = ( np.arange(self.Size[1]) - self.intrinsic[0,2])/self.intrinsic[0,0]
        y_raw[:,0:-1] = np.tile( ( np.arange(self.Size[0]) - self.intrinsic[1,2])/self.intrinsic[1,1],(1,1)).transpose()
        # multiply point by point d_pos and raw matrices
        x = d_pos * x_raw
        y = d_pos * y_raw
        self.Vtx = np.dstack((x, y,d))
    
                
    ##### Compute normals
    def NMap(self):
        self.Nmls = np.zeros(self.Size, np.float32)
        for i in range(1,self.Size[0]-1):
            for j in range(1, self.Size[1]-1):
                nmle1 = normalized_cross_prod(self.Vtx[i+1, j]-self.Vtx[i, j], self.Vtx[i, j+1]-self.Vtx[i, j])
                nmle2 = normalized_cross_prod(self.Vtx[i, j+1]-self.Vtx[i, j], self.Vtx[i-1, j]-self.Vtx[i, j])
                nmle3 = normalized_cross_prod(self.Vtx[i-1, j]-self.Vtx[i, j], self.Vtx[i, j-1]-self.Vtx[i, j])
                nmle4 = normalized_cross_prod(self.Vtx[i, j-1]-self.Vtx[i, j], self.Vtx[i+1, j]-self.Vtx[i, j])
                nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
                if (LA.norm(nmle) > 0.0):
                    nmle = nmle/LA.norm(nmle)
                self.Nmls[i, j] = (nmle[0], nmle[1], nmle[2])
                
    def NMap_optimize(self):
        self.Nmls = np.zeros(self.Size, np.float32)        
        nmle1 = normalized_cross_prod_optimize(self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])        
        nmle2 = normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,2:self.Size[1]  ] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle3 = normalized_cross_prod_optimize(self.Vtx[0:self.Size[0]-2][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle4 = normalized_cross_prod_optimize(self.Vtx[1:self.Size[0]-1][:,0:self.Size[1]-2] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1], \
                                               self.Vtx[2:self.Size[0]  ][:,1:self.Size[1]-1] - self.Vtx[1:self.Size[0]-1][:,1:self.Size[1]-1])
        nmle = (nmle1 + nmle2 + nmle3 + nmle4)/4.0
        norm_mat_nmle = np.sqrt(np.sum(nmle*nmle,axis=2))
        norm_mat_nmle = in_mat_zero2one(norm_mat_nmle)
        #norm division 
        nmle = division_by_norm(nmle,norm_mat_nmle)
        self.Nmls[1:self.Size[0]-1][:,1:self.Size[1]-1] = nmle

    def Draw(self, Pose, s, color = 0) :
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


    def Draw_optimize(self, Pose, s, color = 0) :   
        result = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        stack_pix = np.ones((self.Size[0], self.Size[1]), dtype = np.float32)
        stack_pt = np.ones((np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1)), dtype = np.float32)
        pix = np.zeros((self.Size[0], self.Size[1],2), dtype = np.float32)
        pix = np.dstack((pix,stack_pix))
        pt = np.dstack((self.Vtx[ ::s, ::s, :],stack_pt))
        pt = np.dot(Pose,pt.transpose(0,2,1)).transpose(1,2,0)
        nmle = np.zeros((self.Size[0], self.Size[1],self.Size[2]), dtype = np.float32)
        nmle[ ::s, ::s,:] = np.dot(Pose[0:3,0:3],self.Nmls[ ::s, ::s,:].transpose(0,2,1)).transpose(1,2,0)
        #if (pt[2] != 0.0):
        lpt = np.dsplit(pt,4)
        lpt[2] = in_mat_zero2one(lpt[2])
        # if in 1D pix[0] = pt[0]/pt[2]
        pix[ ::s, ::s,0] = (lpt[0]/lpt[2]).reshape(np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1))
        # if in 1D pix[1] = pt[1]/pt[2]
        pix[ ::s, ::s,1] = (lpt[1]/lpt[2]).reshape(np.size(self.Vtx[ ::s, ::s,:],0), np.size(self.Vtx[ ::s, ::s,:],1))
        pix = np.dot(self.intrinsic,pix[0:self.Size[0],0:self.Size[1]].transpose(0,2,1)).transpose(1,2,0)
        column_index = (np.round(pix[:,:,0])).astype(int)
        line_index = (np.round(pix[:,:,1])).astype(int)
        # create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < self.Size[1])
        cdt_line = (line_index > -1) * (line_index < self.Size[0])
        line_index = line_index*cdt_line
        column_index = column_index*cdt_column
        if (color == 0):
            result[line_index[:][:], column_index[:][:]]= np.dstack((self.color_image[ ::s, ::s,2], \
                                                                     self.color_image[ ::s, ::s,1]*cdt_line, \
                                                                     self.color_image[ ::s, ::s,0]*cdt_column) )
        else:
            result[line_index[:][:], column_index[:][:]]= np.dstack( ( (nmle[ :, :,0]+1.0)*(255./2.), \
                                                                       ((nmle[ :, :,1]+1.0)*(255./2.))*cdt_line, \
                                                                       ((nmle[ :, :,2]+1.0)*(255./2.))*cdt_column ) ).astype(int)
        return result
    
    
        
##################################################################
###################Bilateral Smooth Funtion#######################
##################################################################
    def BilateralFilter(self, d, sigma_color, sigma_space):
        self.depth_image = cv2.bilateralFilter(self.depth_image, d, sigma_color, sigma_space)
        self.skel = cv2.bilateralFilter(self.skel, d, sigma_color, sigma_space)
    



##################################################################
################### Segmentation Function #######################
##################################################################
    def RemoveBG(self,binaryImage):
        ''' This function delete all the little group unwanted from the binary image'''
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

    def GetBody(self,binaryImage,pos2D):
        ''' This function delete all the little group unwanted from the binary image'''
        labeled, n = spm.label(binaryImage)
        threshold = labeled[pos2D[1,1],pos2D[1,0]]
        labeled = (labeled==threshold)
        return labeled
    
    def EntireBdy(self):
        '''this function threshold the depth image in order to to get the whole body alone'''
        pos2D = self.pos2d[0][self.Index]
        max_value = np.iinfo(np.uint16).max # = 65535 for uint16
        tmp = self.depth_image*max_value
        self.depth_image = tmp.astype(np.uint16)
        
        # Threshold according to detph of the body
        bdyVals = self.depth_image[pos2D[:,0]-1,pos2D[:,1]-1]
        #only keep values different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini = np.min(bdy)
        print "mini: %u" % (mini)
        maxi = np.max(bdy)
        print "max: %u" % (maxi)
        bwmin = (self.depth_image > mini+0.08*mini)#0.22*max_value) 
        bwmax = (self.depth_image < maxi-0.35*maxi)#0.3*max_value)
        bw0 = bwmin*bwmax
        # Compare with thenoised binary image given by the kinect
        thresh2,tmp = cv2.threshold(self.bw[0,self.Index],0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = tmp *bw0 
        # Remove all stand alone object
        res = ( self.RemoveBG(res)>0)
        return res#bw0#tmp#
    
    def EntireBdyBB(self):
        '''this function threshold the depth image in order to to get the whole body alone with the bounding box (BB)'''
        pos2D = self.BBBPos
        max_value = np.iinfo(np.uint16).max # = 65535 for uint16
        tmp = self.BBBox*max_value
        self.BBBox = tmp.astype(np.uint16)
        
        # Threshold according to detph of the body
        bdyVals = self.BBBox[pos2D[self.connection[:,0]-1,1]-1,pos2D[self.connection[:,0]-1,0]-1]
        #only keep vales different from 0
        bdy = bdyVals[np.nonzero(bdyVals != 0)]
        mini =  np.min(bdy)
        print "mini: %u" % (mini)
        maxi = np.max(bdy)
        print "max: %u" % (maxi)
        bwmin = (self.BBBox > mini-0.01*max_value) 
        bwmax = (self.BBBox < maxi+0.01*max_value)
        bw0 = bwmin*bwmax
        # Compare with thenoised binary image given by the kinect
        thresh2,tmp = cv2.threshold(self.BBbw,0,1,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        res = tmp * bw0        
        # Remove all stand alone object
        bw0 = ( self.RemoveBG(bw0)>0)

        return res


    def BodySegmentation(self):
        '''this function calls the function in segmentation.py to process the segmentation of the body'''
#==============================================================================
#         self.segm = segm.Segmentation(self.lImages[0,self.Index],self.pos2d[0,self.Index])
#         segImg = (np.zeros([self.Size[0],self.Size[1],self.Size[2],self.numbImages])).astype(np.int8)
#         bdyImg = (np.zeros([self.Size[0],self.Size[1],self.Size[2],self.numbImages])).astype(np.int8) 
#         I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
#==============================================================================
        #Bounding box version
        self.segm = segm.Segmentation(self.BBBox,self.BBBPos) 
        segImg = (np.zeros([self.BBBox.shape[0],self.BBBox.shape[1],self.Size[2],self.numbImages])).astype(np.int8)
        bdyImg = (np.zeros([self.BBBox.shape[0],self.BBBox.shape[1],self.Size[2],self.numbImages])).astype(np.int8) 
        I =  (np.zeros([self.BBBox.shape[0],self.BBBox.shape[1]])).astype(np.int8)
        start_time = time.time()
        #segmentation of the whole body 
        imageWBG = (self.EntireBdyBB()>0)

        #B = self.lImages[0][self.Index]
        B = self.BBBox
        
        right = 0
        left = 1
        armLeft = self.segm.armSeg(imageWBG,B,left)
        armRight = self.segm.armSeg(imageWBG,B,right)
        legRight = self.segm.legSeg(imageWBG,right)
        legLeft = self.segm.legSeg(imageWBG,left)
        head = self.segm.headSeg(imageWBG)
        
        tmp = armLeft[0]+armLeft[1]+armRight[0]+armRight[1]+legRight[0]+legRight[1]+legLeft[0]+legLeft[1]+head
        
        # Visualize the body
        #M = np.max(self.depth_image)


        binaryImage =((imageWBG-(tmp>0))>0)


        #body = ( self.GetBody( binaryImage,self.pos2d[0,self.Index])>0)
        body = ( self.GetBody( binaryImage,self.BBBPos)>0)
        #pdb.set_trace()
        
        bdyImg[:,:,0,self.Index]=body*255#self.depth_image*(255./M)#
        bdyImg[:,:,1,self.Index]=body*255#self.depth_image*(255./M)#
        bdyImg[:,:,2,self.Index]=body*255#self.depth_image*(255./M)#
        return bdyImg[:,:,:,self.Index]
        '''
        correspondance between number and body parts and color
        self.binBody[0] = forearmL      color=[0,0,255]
        self.binBody[1] = upperarmL     color=[200,200,255]
        self.binBody[2] = forearmR      color=[0,255,0]
        self.binBody[3] = upperarmR     color=[200,255,200]
        self.binBody[4] = thighR        color=[255,0,255]
        self.binBody[5] = calfR         color=[255,180,255]
        self.binBody[6] = thighL        color=[255,255,0]
        self.binBody[7] = calfL         color=[255,255,180]
        self.binBody[8] = headB         color=[255,0,0]
        self.binBody[9] = body          color=[255,255,255] 
        '''
        
        # For Channel color R
        I = I +0*armLeft[0]
        I = I +200*armLeft[1]
        I = I +0*armRight[0]
        I = I +200*armRight[1]
        I = I +255*legRight[0]
        I = I +255*legRight[1]
        I = I +255*legLeft[0]
        I = I +255*legLeft[1]
        I = I +255*head
        I = I +255*body
        segImg[:,:,0,self.Index]=I
    
        # For Channel color G
        #I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
        I =  (np.zeros([self.BBBox.shape[0],self.BBBox.shape[1]])).astype(np.int8)
        I = I +0*armLeft[0]
        I = I +200*armLeft[1]
        I = I +255*armRight[0]
        I = I +255*armRight[1]
        I = I +0*legRight[0]
        I = I +180*legRight[1]
        I = I +255*legLeft[0]
        I = I +255*legLeft[1]
        I = I +0*head
        I = I +255*body
        segImg[:,:,1,self.Index] = I
    
        # For Channel color B
        #I =  (np.zeros([self.Size[0],self.Size[1]])).astype(np.int8)
        I =  (np.zeros([self.BBBox.shape[0],self.BBBox.shape[1]])).astype(np.int8)
        I = I +255*armLeft[0]
        I = I +255*armLeft[1]
        I = I +0*armRight[0]
        I = I +200*armRight[1]
        I = I +255*legRight[0]
        I = I +255*legRight[1]
        I = I +0*legLeft[0]
        I = I +180*legLeft[1]
        I = I +0*head
        I = I +255*body
        segImg[:,:,2,self.Index] = I
    
        elapsed_time = time.time() - start_time
        print "Segmentation: %f" % (elapsed_time)
        return segImg[:,:,:,self.Index]

    
###################################################################
################### Bounding boxes Function #######################
##################################################################      
    def BodyBBox(self):       
        '''This will generate a new depthframe but focuses on the human body'''
        pos2D = self.pos2d[0,self.Index].astype(np.int16)
        # extremes points of the bodies
        minV = np.min(pos2D[:,1])
        maxV = np.max(pos2D[:,1])
        minH = np.min(pos2D[:,0])
        maxH = np.max(pos2D[:,0])
        # distance head to neck. Let us assume this is enough for all borders
        distH2N = int(LA.norm(pos2D[self.connection[0,1]-1]-pos2D[self.connection[0,0]-1]))
        Box = self.lImages[0,self.Index]
        bwBox = self.bw[0,self.Index]
        ############ Should check whether the value are in the frame #####################
        self.BBBox = Box[minV-distH2N:maxV,minH-distH2N:maxH+distH2N]
        self.BBBPos = (pos2D -np.array([minH-distH2N,minV-distH2N])).astype(np.int16)
        self.BBbw = bwBox[minV-distH2N:maxV,minH-distH2N:maxH+distH2N]
        
        
#==============================================================================
#         # threshold according to the position
#         minIdxV = np.argmin(pos2D[self.connection[:,0]-1,1])
#         maxIdxV = np.argmax(pos2D[self.connection[:,0]-1,1])
#         minIdxH = np.argmin(pos2D[self.connection[:,0]-1,0])
#         maxIdxH = np.argmax(pos2D[self.connection[:,0]-1,0])
#         
#         for i in range(minIdxV):
#         
#==============================================================================
        
        
        
        
        
        
        
        
        
        
