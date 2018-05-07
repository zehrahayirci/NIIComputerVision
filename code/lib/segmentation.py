"""
File created by Inoe ANDRE the 01-03-2017
Define functions to do the segmentation in a depthmap image
"""
import cv2
import numpy as np
from numpy import linalg as LA
import imp
import scipy as sp
import scipy.ndimage
import math
import time
import itertools
import scipy.ndimage.measurements as spm


'''These are the order of joints returned by the kinect adaptor.
    SpineBase = 0
    SpineMid = 1
    Neck = 2
    Head = 3
    ShoulderLeft = 4
    ElbowLeft = 5
    WristLeft = 6
    HandLeft = 7
    ShoulderRight = 8
    ElbowRight = 9
    WristRight = 10
    HandRight = 11
    HipLeft = 12
    KneeLeft = 13
    AnkleLeft = 14
    FootLeft = 15
    HipRight = 16
    KneeRight = 17
    AnkleRight = 18
    FootRight = 19
    SpineShoulder = 20
    HandTipLeft = 21
    ThumbLeft = 22
    HandTipRight = 23
    ThumbRight = 24
   '''

class Segmentation(object):
    """
    Segmentation process concerning body parts
    """
    def __init__(self, depthImage, pos2D):
        """
        Constructor
        :param depthImage: Cropped depth image of the current image
        :param pos2D: list of position of the junction
        """
        self.depthImage = depthImage
        self.pos2D = pos2D
        self.bodyPts = []


    def findSlope(self,A,B):
        """
        Get the slope of a line made from two point A and B or the distance in one axes
        :param A: point 1
        :param B: point 2
        :return: an array of coefficient
        a is the normalized distance in the x axis
        b is the normalized distance in the y axis
        c is the constant between the two points
        """
        #Be sure A and B are different
        if (A == B).all():
            print "There is no slope between a point and itself"
            return np.array([0.0,0.0,0.0])
        A = A.astype(np.float32)
        B = B.astype(np.float32)
        # distance in Y axis
        diffY = B[1]-A[1]
        # distance in X axis
        diffX = A[0]-B[0]
        dist = np.sqrt(np.square(diffY) + np.square(diffX))
        a = diffY/dist # normalized distance
        b = diffX/dist # normalized distance
        #constant in this line
        c = -a*A[0]-b*A[1]
        return np.array([a,b,c])

    def inferedPoint(self,A,a,b,c,point,T=100):
        """
        Find two points that are the corners of the segmented part
        :param A: Depth Image
        :param a: dist x axe between two points
        :param b: dist y axe between two points
        :param c: constant of this line
        :param point: a junction
        :param T: max distance to find intersection
        :return: two intersection points between a slope and the edges of the body part
        """
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        process_y = abs(a) > abs(b)
        # searching in y axis
        if process_y:
            y = int(point[1])
            # search an edge running through a slope with decreasing y
            while 1:
                y = y-1
                # keep track of the perpendicular slope
                x = int(np.round(-(b*y+c)/a))
                inImage = (x>=0) and (x<col) and (y>=0) and (y<line)
                if(inImage):
                    # if an edge is reached
                    if A[y,x]==0:
                        x_up = x
                        y_up = y
                        break
                    else:
                        # if the max distance is reached
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T:
                            x_up = x
                            y_up = y
                            break
                else:
                    y_up = y+1
                    x_up = int(np.round(-(b*y+c)/a))
                    break
            y = int(point[1])
            # search an edge running through a slope with increasing y
            while 1:
                y = y+1
                # keep track of the perpendicular slope
                x = int(np.round(-(b*y+c)/a))
                inImage = (x>=0) and (x<col) and (y>=0) and (y<line)
                if(inImage):
                    # if an edge is reached
                    if A[y,x]==0:
                        x_down = x
                        y_down = y
                        break
                    else:
                        # if the max distance is reached
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#math.sqrt((x-point(1))^2+(y-point(2))^2)>T:
                            x_down = x
                            y_down = y
                            break
                else:
                    y_down = y-1
                    x_down = int(np.round(-(b*y+c)/a))
                    break
            if x_up>x_down:
                right = [x_up, y_up]
                left = [x_down, y_down]
            else:
                left = [x_up, y_up]
                right = [x_down, y_down]
        # searching in x axis
        else:#process_x
            x = int(point[0])
            while 1:
                x = x-1
                # keep the track of the perpendicular slope
                y = int(np.round(-(a*x+c)/b))
                inImage = (x>=0) and (x<col) and (y>=0) and (y<line)
                if inImage:
                    # if an edge is reached
                    if A[int(y),int(x)]==0:
                        x_left = x
                        y_left = y
                        break
                    else:
                        # if the max distance is reached
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T
                            x_left = x
                            y_left = y
                            break
                else:
                    x_left = x+1
                    y_left = int(np.round(-(a*x_left+c)/b))
                    break

            x = int(point[0])
            while 1:
                x = x+1
                # keep the track of the perpendicular slope
                y = int(np.round(-(a*x+c)/b))
                inImage = (x>=0) and (x<col) and (y>=0) and (y<line)
                if inImage:
                    # if an edge is reached
                    if A[int(y),int(x)]==0:
                        x_right = x
                        y_right = y
                        break
                    else:
                        # if the max distance is reached
                        distCdt = LA.norm([x,y]-point)>T
                        if distCdt:#sqrt((x-point(1))^2+(y-point(2))^2)>T
                            x_right = x
                            y_right = y
                            break
                else:
                    x_right = x-1
                    y_right = int(np.round(-(a*x_right+c)/b))
                    break
            left = [x_left, y_left]
            right = [x_right, y_right]
        return [left, right]



    def polygon(self,slopes,ref,  limit  ):
        """
        Test the sign of alpha = (a[k]*j+b[k]*i+c[k])*ref[k]
        to know whether a point is within a polygon or not
        :param slopes: list of slopes defining a the border lines of the polygone
        :param ref:  a point inside the polygon
        :param limit: number of slopes
        :return: the body part filled with true.
        """
        start_time = time.time()
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        res = np.zeros([line,col],np.bool)
        alpha = np.zeros([1,limit])
        # for each point in the image
        for i in range(line):
           for j in range(col):
               for k in range(limit):
                   # compare distance of the point to a slope
                   alpha[0][k] = (slopes[0][k]*j+slopes[1][k]*i+slopes[2][k])*ref[0,k]
               alpha_positif = (alpha >= 0)
               if alpha_positif.all():
                   res[i,j]=True
        elapsed_time = time.time() - start_time
        print "polygon: %f" % (elapsed_time)
        return res

    def polygon_optimize(self,slopes,ref,  limit  ):
        """
        Test the sign of alpha = (a[k]*j+b[k]*i+c[k])*ref[k]
        to know whether a point is within a polygon or not
        :param slopes: list of slopes defining a the border lines of the polygone
        :param ref:  a point inside the polygon
        :param limit: number of slopes
        :return: the body part filled with true.
        """
        #start_time = time.time()
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        res = np.ones([line,col])

        #create a matrix containing in each pixel its indices
        lineIdx = np.array([np.arange(line) for _ in range(col)]).transpose()
        colIdx = np.array([np.arange(col) for _ in range(line)])
        depthIdx = np.ones([line,col])
        ind = np.stack( (colIdx,lineIdx,depthIdx), axis = 2)
        alpha = np.zeros([line,col,limit])
        alpha= np.dot(ind,slopes)
        # for each k (line) if the points (ref and the current point in alpha) are on the same side then the operation is positive
        for k in range(limit):
            alpha[:,:,k]=( (np.dot(alpha[:,:,k],ref[0][k])) >= 0)
        # make sure that each point are on the same side as the reference for all line of the polygon
        for k in range(limit):
            res = res*alpha[:,:,k ]
        #threshold the image so that only positiv values (same side as reference point) are kept.
        #res = (res>0)
        #elapsed_time = time.time() - start_time
        #print "polygon_optimize: %f" % (elapsed_time)
        return res

    def polygonOutline(self,points):
        """
        Find a polygon on the image through the points given in points
        :param points: array of points which are the corners of the polygon to find
        :return:  the body part filled with true.
        """

        #check if there is repeat point in points
        j=0
        for i in range(1, points.shape[0]):
            if(points[i,0]!=points[j,0] or points[i,1]!=points[j,1]):
                j=j+1
                points[j] = points[i]
        points = points[0:j+1]
        if(points[0,0]==points[-1,0] and points[0,1]==points[-1,1]):
            points = points[0:-1]
            j=j-1
        if(j!=i):
            print("there is repeat point in the polygon")

        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        im_out = np.zeros([line,col],np.uint8)
        points = points.astype(np.float64)
        n = points.shape[0]
        i = 2
        d = 0

        # copy the point but with a circular permutation
        ptB = np.zeros(points.shape)
        ptB[-1]=points[0]
        for i in range(0,points.shape[0]-1):
            ptB[i] = points[i+1]
        # trace the segment
        M = np.zeros([line,col],np.uint8)
        for i in range(n-d):
            A = points[i,:]
            B = ptB[i,:]
            slopes = self.findSlope(A,B)
            # if dist in x is longer than dist in y
            if np.abs(slopes[0]) > np.abs(slopes[1]):
                # if Ay have a higher value than By permute A and B
                if A[1] > B[1]:
                    tmp = B
                    B = A
                    A = tmp
                # trace the slope between the two points
                for y in range(int(A[1]),int(B[1])+1):
                    x = np.round(-(slopes[1]*y+slopes[2])/slopes[0])
                    M[y,int(x)]= 1
            else :
                # if Ax have a higher value than Bx permute A and B
                if A[0] > B[0]:
                    tmp = B
                    B = A
                    A = tmp
                # trace the slope between the two points
                for x in range(int(A[0]),int(B[0])+1):
                    y = np.round(-(slopes[0]*x+slopes[2])/slopes[1])
                    M[int(y),x]= 1
        ## Fill the polygon
        # Copy the thresholded image.
        im_floodfill = M.copy()
        im_floodfill = im_floodfill.astype(np.uint8)

        # Mask used to flood filling.
        # Notice the size needs to be 2 pixels than the image.
        h, w = M.shape[:2]
        mask = np.zeros((h+2, w+2), np.uint8)

        # Floodfill from point (0, 0)
        cv2.floodFill(im_floodfill, mask, (0,0), 255)

        # Invert floodfilled image
        im_floodfill_inv = cv2.bitwise_not(im_floodfill)

        # Combine the two images to get the foreground.
        im_out = M | im_floodfill_inv
        return im_out>0

    def nearestPeak(self,A,hipLeft,hipRight,knee_right, spine):
        """
        In the case of upper legs, find the point in between the two upper legs that is at a edge of the hip
        :param A: binary image
        :param hipLeft: left hip junctions
        :param hipRight:  right hip junctions
        :param knee_right: right knee junctions
        :return: return a point at the edge and between the two legs
        Make drawing will help to understand
        """
        # check which hip is lower
        if (int(hipLeft[0])<int(hipRight[0])):
            # check which hip is lower
            # extract rectangle from the tree points depending on each other position
            if (int(hipLeft[1])<int(knee_right)):
                region = A[int(hipLeft[1]):int(knee_right),int(hipLeft[0]):int(hipRight[0])]
                pt_start = [int(hipLeft[0]),int(hipLeft[1])]
            else:
                print("has met yet line357")
                exit()
                region = A[int(knee_right):int(hipLeft[1]),int(hipLeft[0]):int(hipRight[0])]
                pt_start = [int(hipLeft[0]),int(knee_right)]
        else:
            # check which hip is lower
            # extract rectangle from the tree points depending on each other position
            if (int(hipRight[1])<int(knee_right)):
                print("has met yet line365")
                exit()
                region = A[int(hipLeft[1]):int(knee_right),int(hipRight[0]):int(hipLeft[0])]
                pt_start = [int(hipRight[0]),int(hipLeft[1])]
            else:
                print("has met yet line370")
                exit()
                region = A[int(knee_right):int(hipLeft[1]),int(hipRight[0]):int(hipLeft[0])]
                pt_start = [int(hipRight[0]),int(knee_right)]
        f = np.nonzero( (region==0) )
        if(sum(sum(f))==0):
            print("there is no hole between two upper legs")
            return np.array([0,0])
        # Get the highest point among the point that not in the body
        #d = np.argmin(f[0])
        # Get the closest point to the spine
        d = np.argmin(np.sum( np.square(np.array([spine[0]-f[1]+1-pt_start[0], spine[1]-f[0]+1-pt_start[1]]).transpose()),axis=1 ))
        return np.array([f[1][d]-1+pt_start[0],f[0][d]-1+pt_start[1]])

    def rearmSeg(self, A, side):
        """
        resegment the arm into two body parts
        :param A: depthImag
        :param side: if side = 0 the segmentation will be done for the right arm
                  otherwise it will be for the left arm
        :return: an array containing two body parts : an upper arm and a lower arm
        """
        # junction position (-1 adapted for python)
        pos2D = self.pos2D.astype(np.float64)-1
        # Right arm
        if side == 0 :
            shoulder =8
            elbow = 9
            wrist = 10
            foreArmPts = self.foreArmPtsR
            intersection_shoulder = self.upperArmPtsR[1]
            peakArmpit = self.upperArmPtsR[2]
        # Left arm
        else :
            shoulder =4
            elbow = 5
            wrist = 6
            foreArmPts = self.foreArmPtsL
            intersection_shoulder = self.upperArmPtsL[1]
            peakArmpit = self.upperArmPtsL[2]

        ## lower arm
        # FindSlopes give the slope of a line made by two points
        slopesElbow = self.findSlope(foreArmPts[0], foreArmPts[1])
        a_pen = slopesElbow[0]
        b_pen = slopesElbow[1]
        c_pen = slopesElbow[2]
        slopesWrist = self.findSlope(foreArmPts[2], foreArmPts[3])
        a_pen67 = slopesWrist[0]
        b_pen67 = slopesWrist[1]
        c_pen67 = slopesWrist[2]

        # find lenght of arm
        bone1 = LA.norm(pos2D[elbow]-pos2D[wrist])
        bone2 = LA.norm(pos2D[elbow]-pos2D[shoulder])
        bone = max(bone1,bone2)

        # compute the intersection between the slope and the extremety of the body
        intersection_elbow=self.inferedPoint(A,a_pen,b_pen,c_pen,foreArmPts[0]/2+foreArmPts[1]/2,0.5*bone/1.6)
        vect_elbow = intersection_elbow[0]-pos2D[elbow]
        intersection_wrist=self.inferedPoint(A,a_pen67,b_pen67,c_pen67,foreArmPts[2]/2+foreArmPts[3]/2,bone/2/2)
        vect_wrist = intersection_wrist[0]-pos2D[wrist]
        vect67 = pos2D[wrist]-pos2D[elbow]
        vect67_pen = np.array([vect67[1], -vect67[0]])
        # reorder points if necessary
        if sum(vect67_pen*vect_elbow)*sum(vect67_pen*vect_wrist)<0:
            print("have never met line444")
            x = intersection_elbow[0]
            intersection_elbow[0] = intersection_elbow[1]
            intersection_elbow[1] = x
            vect_elbow = intersection_elbow[0]-pos2D[elbow]

        # list of the 4 points defining the corners the forearm
        pt4D = np.array([intersection_elbow[0],intersection_elbow[1],intersection_wrist[1],intersection_wrist[0]])
        # list of the 4 points defining the corners the forearm permuted
        pt4D_bis = np.array([intersection_wrist[0],intersection_elbow[0],intersection_elbow[1],intersection_wrist[1]])
        if side == 0 :
            self.foreArmPtsR = pt4D
        else:
            self.foreArmPtsL = pt4D
        # Get slopes for each line of the polygon
        finalSlope=self.findSlope(pt4D.transpose(),pt4D_bis.transpose())
        x = np.isnan(finalSlope[0])
        if sum(x)!=0:
            print("have never met line468")
            exit()
        #erase all NaN in the array
        polygonSlope = np.zeros([3,finalSlope[0][~np.isnan(finalSlope[0])].shape[0]])
        polygonSlope[0]=finalSlope[0][~np.isnan(finalSlope[0])]
        polygonSlope[1]=finalSlope[1][~np.isnan(finalSlope[1])]
        polygonSlope[2]=finalSlope[2][~np.isnan(finalSlope[2])]
        # get reference point
        midpoint = [(pos2D[elbow,0]+pos2D[wrist,0])/2, (pos2D[elbow,1]+pos2D[wrist,1])/2]
        ref= np.array([polygonSlope[0]*midpoint[0] + polygonSlope[1]*midpoint[1] + polygonSlope[2]]).astype(np.float32)
        #fill the polygon
        bw_up = ( A*self.polygon_optimize(polygonSlope,ref,x.shape[0]-sum(x)))


        ## upper arm
        # FindSlopes give the slope of a line made by two points
        slopesshoulderpeak = self.findSlope(intersection_shoulder, peakArmpit)
        a_pen = slopesshoulderpeak[0]
        b_pen = slopesshoulderpeak[1]
        c_pen = slopesshoulderpeak[2]

        # compute the intersection between the slope and the extremety of the body
        intersection_shoulderpeak=self.inferedPoint(A,a_pen,b_pen,c_pen,intersection_shoulder/2+peakArmpit/2,0.5*bone)
        if LA.norm(intersection_shoulderpeak[0]-intersection_shoulder)<LA.norm(intersection_shoulderpeak[1]-intersection_shoulder):
            intersection_shoulder = intersection_shoulderpeak[0]
            peakArmpit = intersection_shoulderpeak[1]
        else:
            intersection_shoulder = intersection_shoulderpeak[1]
            peakArmpit = intersection_shoulderpeak[0]

        #find peakshoulder
        slopesshoulder = self.findSlope(self.peakshoulderL, self.peakshoulderR)
        a_pen = slopesshoulder[0]
        b_pen = slopesshoulder[1]
        c_pen = slopesshoulder[2]
        intersection_peakshoulder=self.inferedPoint(A,a_pen,b_pen,c_pen,self.peakshoulderL/2+self.peakshoulderR/2)
        if side==0:
            peakshoulder = intersection_peakshoulder[1]
        else:
            peakshoulder = intersection_peakshoulder[0]

        # check if intersection is on the head
        if intersection_shoulder[1]<pos2D[3][1]:
            print("intersection shoulder is upper the head")
            intersection_shoulder[1] = pos2D[2][1]
            intersection_shoulder[0] = np.round(-(b_pen*intersection_shoulder[1]+c_pen)/a_pen)

        # constraint on peakArmpit
        if side == 0 and peakArmpit[0]>intersection_elbow[0][0]:
            print "meet the constrains on peakArmpitR"
            peakArmpit = self.upperArmPtsR[2]
        elif side==1 and peakArmpit[0]<intersection_elbow[1][0]:
            print "meet the constrains on peakArmpitL"
            peakArmpit = self.upperArmPtsL[2]

        # check if intersection is on the head
        if peakshoulder[1]<pos2D[2][1]:
            temp = peakshoulder
            peakshoulder[1] = pos2D[2][1]
            slopesPeakShoulder = self.findSlope(np.array(temp),np.array(peakArmpit))
            if(side==0):
                print("peakshoulder is upper the neck R")
                peakshoulder[0] = np.round(-(slopesPeakShoulder[1]*peakshoulder[1]+slopesPeakShoulder[2])/slopesPeakShoulder[0])
            else:
                print("peakshoulder is upper the neck L")
                peakshoulder[0] = np.round(-(slopesPeakShoulder[1]*peakshoulder[1]+slopesPeakShoulder[2])/slopesPeakShoulder[0])

        #cross product to know which point to select
        vect65 = pos2D[shoulder]-pos2D[elbow]
        t = np.cross(np.insert(vect_elbow, vect_elbow.shape[0],0),np.insert(vect65, vect65.shape[0],0))
        if t[2]>0:
            tmp = intersection_elbow[0]
            intersection_elbow[0] = intersection_elbow[1]
            intersection_elbow[1] = tmp
            print("line504")

        # create the upperarm polygon out the five point defining it
        if side != 0 :
            ptA = np.stack((intersection_elbow[1],intersection_shoulder,peakArmpit,intersection_elbow[0]))
            self.upperArmPtsL = ptA
            self.peakshoulderL = np.array(peakshoulder).astype(np.int32)
        else:
            ptA = np.stack((intersection_elbow[0],intersection_shoulder,peakArmpit,intersection_elbow[1]))
            self.upperArmPtsR = ptA
            self.peakshoulderR = np.array(peakshoulder).astype(np.int32)

        bw_upper = (A*self.polygonOutline(ptA))

        return np.array([bw_up,bw_upper])

    def armSeg(self,A,B,side):
        """
        Segment the left arm into two body parts
        :param A: depthImag
        :param B: depthImg after bilateral filtering
        :param side: if side = 0 the segmentation will be done for the right arm
                  otherwise it will be for the left arm
        :return: an array containing two body parts : an upper arm and a lower arm
        """

        # pos2D[4] = Shoulder_Left
        # pos2D[5] = Elbow_Left
        # pos2D[6] = Wrist_Left
        # pos2D[8] = Shoulder_Right
        # pos2D[9] = Elbow_Right
        # pos2D[10] = Wrist_Right

        # junction position (-1 adapted for python)
        pos2D = self.pos2D.astype(np.float64)-1
        # Right arm
        if side == 0 :
            shoulder =8
            elbow = 9
            wrist = 10
        # Left arm
        else :
            shoulder =4
            elbow = 5
            wrist = 6


        # First let us see the down limit thanks to the elbow and the wrist

        # FindSlopes give the slope of a line made by two points
        # Forearm
        slopesForearm=self.findSlope(pos2D[elbow],pos2D[wrist])
        a_pen67 = -slopesForearm[1]
        b_pen67 = slopesForearm[0]
        # Upperarm
        slopesUpperarm=self.findSlope(pos2D[elbow],pos2D[shoulder])
        a_pen = slopesForearm[0] + slopesUpperarm[0]
        b_pen = slopesForearm[1] + slopesUpperarm[1]
        if a_pen * b_pen == 0:
            a_pen = slopesUpperarm[1]
            b_pen =-slopesUpperarm[0]

        # Perpendicular slopes
        c_pen = -(a_pen*pos2D[elbow,0]+b_pen*pos2D[elbow,1])


        # find lenght of arm
        bone1 = LA.norm(pos2D[elbow]-pos2D[wrist])
        bone2 = LA.norm(pos2D[elbow]-pos2D[shoulder])
        bone = max(bone1,bone2)

        # compute the intersection between the slope and the extremety of the body
        # And get two corners of the segmented body parts
        intersection_elbow=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[elbow],0.5*bone/1.6)
        vect_elbow = intersection_elbow[0]-pos2D[elbow]

        # Slope forearm
        c_pen67=-(a_pen67*pos2D[wrist,0]+b_pen67*pos2D[wrist,1])
        # get intersection near the wrist
        intersection_wrist=self.inferedPoint(A,a_pen67,b_pen67,c_pen67,pos2D[wrist],bone/2/2)
        #intersection_wrist=self.inferedPoint(A,a_pen67,b_pen67,c_pen67,pos2D[wrist],bone/2)# MIT
        vect_wrist = intersection_wrist[0]-pos2D[wrist]
        vect67 = pos2D[wrist]-pos2D[elbow]
        vect67_pen = np.array([vect67[1], -vect67[0]])
        # reorder points if necessary
        if sum(vect67_pen*vect_elbow)*sum(vect67_pen*vect_wrist)<0:
            print("have never met line449")
            x = intersection_elbow[0]
            intersection_elbow[0] = intersection_elbow[1]
            intersection_elbow[1] = x
            vect_elbow = intersection_elbow[0]-pos2D[elbow]

        # list of the 4 points defining the corners the forearm
        pt4D = np.array([intersection_elbow[0],intersection_elbow[1],intersection_wrist[1],intersection_wrist[0]])
        # list of the 4 points defining the corners the forearm permuted
        pt4D_bis = np.array([intersection_wrist[0],intersection_elbow[0],intersection_elbow[1],intersection_wrist[1]])
        if side == 0 :
            self.foreArmPtsR = pt4D
        else:
            self.foreArmPtsL = pt4D
        # Get slopes for each line of the polygon
        finalSlope=self.findSlope(pt4D.transpose(),pt4D_bis.transpose())
        x = np.isnan(finalSlope[0])
        if sum(x)!=0:
            print("have never met line468")
            exit()
        #erase all NaN in the array
        polygonSlope = np.zeros([3,finalSlope[0][~np.isnan(finalSlope[0])].shape[0]])
        polygonSlope[0]=finalSlope[0][~np.isnan(finalSlope[0])]
        polygonSlope[1]=finalSlope[1][~np.isnan(finalSlope[1])]
        polygonSlope[2]=finalSlope[2][~np.isnan(finalSlope[2])]
        # get reference point
        midpoint = [(pos2D[elbow,0]+pos2D[wrist,0])/2, (pos2D[elbow,1]+pos2D[wrist,1])/2]
        ref= np.array([polygonSlope[0]*midpoint[0] + polygonSlope[1]*midpoint[1] + polygonSlope[2]]).astype(np.float32)
        #fill the polygon
        bw_up = ( A*self.polygon_optimize(polygonSlope,ref,x.shape[0]-sum(x)))

        # pos2D[2] = Neck
        # pos2D[3] = Head

        #compute slopes Neck Head (SH)spine
        slopesSH=self.findSlope(pos2D[2],pos2D[3])
        a_pen = slopesSH[1]
        b_pen = - slopesSH[0]
        c_pen = -(a_pen*pos2D[2,0]+b_pen*pos2D[2,1])

        # compute the intersection between the slope and the extremety of the body
        intersection_head=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[2])

        # find the peak of shoulder
        points = np.zeros([5,2])
        points[0:4,:] = pos2D[[elbow, shoulder, 20, 3],:]
        points[4, :] = [pos2D[elbow,0], pos2D[3][1]]
        B1 = np.logical_and( (A==0),self.polygonOutline(points))
        f = np.nonzero(B1)
        if(sum(sum(f))==0):
            print("there is not hole between shoulder and body")
            peakshoulder = np.array([(pos2D[elbow,0]*2+pos2D[20,0])/3,(pos2D[elbow,1]*2+pos2D[20,1])/3])
        else:
            # find the minimum in distance to shoulder
            d = np.argmin(np.sum( np.square(np.array([pos2D[20,0]-f[1], pos2D[20,1]-f[0]]).transpose()),axis=1 ))
            peakshoulder = np.array([f[1][d],f[0][d]])


        slopesTorso=self.findSlope(pos2D[20],pos2D[shoulder])

        a_pen = slopesTorso[0]+slopesUpperarm[0]
        b_pen = slopesTorso[1]+slopesUpperarm[1]
        if a_pen * b_pen == 0:
            a_pen = slopesTorso[1]
            b_pen = -slopesTorso[0]

        #slope of the shoulder
        c_pen = -(a_pen*pos2D[shoulder,0]+b_pen*pos2D[shoulder,1])


        intersection_shoulder = self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[shoulder])
        vect65 = pos2D[shoulder]-pos2D[elbow]

        #
        vect_215 = intersection_shoulder[0]-pos2D[shoulder]
        #cross product to know which point to select
        t = np.cross(np.insert(vect_elbow, vect_elbow.shape[0],0),np.insert(vect65, vect65.shape[0],0))
        t1 = np.cross(np.insert(vect_215,vect_215.shape[0],0),np.insert(-vect65,vect65.shape[0],0))
        if t1[2]>0:
            tmp = intersection_shoulder[0]
            intersection_shoulder[0] = intersection_shoulder[1]
            intersection_shoulder[1] = tmp
            print("wrong inline515")

        if t[2]<0:
            tmp = intersection_elbow[0]
            intersection_elbow[0] = intersection_elbow[1]
            intersection_elbow[1] = tmp
            print("wrong in line523")

        # check if intersection is on the head
        if(side==0):
            if intersection_shoulder[1][1]<pos2D[3][1]:
                print("intersection shoulder is upper the head R")
                intersection_shoulder[1][1] = pos2D[2][1]
                intersection_shoulder[1][0] = np.round(-(b_pen*intersection_shoulder[1][1]+c_pen)/a_pen)
                peakshoulder = np.array(intersection_shoulder[1])
        else:
            if intersection_shoulder[0][1]<pos2D[3][1]:
                print("intersection shoulder is upper the head L")
                intersection_shoulder[0][1] = pos2D[2][1]
                intersection_shoulder[0][0] = np.round(-(b_pen*intersection_shoulder[0][1]+c_pen)/a_pen)
                peakshoulder = np.array(intersection_shoulder[0])


        # the upper arm need a fifth point -> Let us find it by finding the closest point to shoulder point
        points = np.zeros([5,2])
        points[0:4,:] = pos2D[[elbow, shoulder, 20, 0],:]
        points[4, :] = [pos2D[elbow,0], pos2D[1][1]]
        B1 = np.logical_and( (A==0),self.polygonOutline(points))
        f = np.nonzero(B1)
        if(sum(sum(f))!=0):
            # find the minimum in distance to shoulder
            d = np.argmin(np.sum( np.square(np.array([pos2D[20,0]-f[1], pos2D[20,1]-f[0]]).transpose()),axis=1 ))
            peakArmpit = np.array([f[1][d],f[0][d]])
        else:
            print "there is no hole between the arm and body"
            region = A*0.0
            if side==0:
                region[int(pos2D[20,1]):int(pos2D[0,1]),int(pos2D[20,0]):] = 1
            else:
                region[int(pos2D[20,1]):int(pos2D[0,1]), 0:int(pos2D[20,0])] = 1
            B1 = np.logical_and( (A==0),region)
            f = np.nonzero(B1)
            if(sum(sum(f))!=0):
                # find the minimum in distance to shoulder
                d = np.argmin(np.sum( np.square(np.array([pos2D[20,0]-f[1], pos2D[20,1]-f[0]]).transpose()),axis=1 ))
                peakArmpit = np.array([f[1][d],f[0][d]])
            else:
                print "the peakArmpit is wrong"
                peakArmpit = np.array([pos2D[shoulder,0], pos2D[1,1]])
        # constraint on peakArmpit
        if side == 0 and peakArmpit[0]>intersection_elbow[0][0]:
            print "meet the constrains on peakArmpitR"
            peakArmpit = [intersection_elbow[0][0]/2 + intersection_shoulder[1][0]/2-2, intersection_elbow[0][1]/2 + intersection_shoulder[1][1]/2]
        elif side==1 and peakArmpit[0]<intersection_elbow[1][0]:
            print "meet the constrains on peakArmpitL"
            peakArmpit = [intersection_elbow[1][0]/2 + intersection_shoulder[0][0]/2+2, intersection_elbow[1][1]/2 + intersection_shoulder[0][1]/2]

        # check if intersection is on the head
        if peakshoulder[1]<pos2D[2][1]:
            temp = peakshoulder
            peakshoulder[1] = pos2D[2][1]
            slopesPeakShoulder = self.findSlope(temp,np.array([peakArmpit[0], peakArmpit[1]]))
            if(side==0):
                print("peakshoulder is upper the neck R")
                peakshoulder[0] = np.round(-(slopesPeakShoulder[1]*peakshoulder[1]+slopesPeakShoulder[2])/slopesPeakShoulder[0])
            else:
                print("peakshoulder is upper the neck L")
                peakshoulder[0] = np.round(-(slopesPeakShoulder[1]*peakshoulder[1]+slopesPeakShoulder[2])/slopesPeakShoulder[0])

        # create the upperarm polygon out the five point defining it
        if side != 0 :
            #ptA = np.stack((intersection_elbow[0],intersection_shoulder[0],peakshoulder,peakArmpit,intersection_elbow[1]))
            ptA = np.stack((intersection_elbow[0],intersection_shoulder[0],peakArmpit,intersection_elbow[1]))
            self.upperArmPtsL = ptA
            self.peakshoulderL = peakshoulder
        else:
            #ptA = np.stack((intersection_elbow[1],intersection_shoulder[1],peakshoulder,peakArmpit,intersection_elbow[0]))
            ptA = np.stack((intersection_elbow[1],intersection_shoulder[1],peakArmpit,intersection_elbow[0]))
            self.upperArmPtsR = ptA
            self.peakshoulderR = peakshoulder

        bw_upper = (A*self.polygonOutline(ptA))

        return np.array([bw_up,bw_upper])

    def relegSeg(self, A, side):
        """
        Segment the leg into two body parts
        :param A: depthImag
        :param side: if side = 0 the segmentation will be done for the right leg
                  otherwise it will be for the left leg
        :return: an array containing two body parts : an upper leg and a lower leg
        """

        pos2D = self.pos2D.astype(np.float64)-1

        # Right
        if side == 0 :
            knee =17
            hip = 16
            ankle = 18
            thighPts = self.thighPtsR
            calfPts = self.calfPtsR
        else : # Left
            knee =13
            hip = 12
            ankle = 14
            thighPts = self.thighPtsL
            calfPts = self.calfPtsL

        ## thigh
        # get peak
        peak = self.thighPtsL[0,:]

        # compute slopes related to the leg position
        slopeKnee = self.findSlope(thighPts[2], thighPts[3])
        a_pen = slopeKnee[0]
        b_pen = slopeKnee[1]
        c_pen = slopeKnee[2]

        # find lenght of leg
        bone1 = LA.norm(pos2D[knee]-pos2D[ankle])
        bone2 = LA.norm(pos2D[knee]-pos2D[hip])
        bone = max(bone1,bone2)

        # find 2 points corner of the knee
        intersection_knee=self.inferedPoint(A,a_pen,b_pen,c_pen,thighPts[2]/2+thighPts[3]/2, bone/1.5/2)
        if(side!=0): # if two knees are too close
            if(intersection_knee[1][0]>(pos2D[13][0]+pos2D[17][0])/2):
                print("two knees are too close L")
                intersection_knee[1][0] = (pos2D[13][0]+pos2D[17][0])/2
        else:
            if(intersection_knee[0][0]<(pos2D[13][0]+pos2D[17][0])/2):
                intersection_knee[0][0] = (pos2D[13][0]+pos2D[17][0])/2
                print("two knees are too close R")

        # find right side of the hip rsh
        slopeRsh = self.findSlope(thighPts[0], thighPts[1])
        a_pen = slopeRsh[0]
        b_pen = slopeRsh[1]
        c_pen = slopeRsh[2]
        # find 2 points corner of the knee
        intersection_rsh=self.inferedPoint(A,a_pen,b_pen,c_pen,thighPts[0]/2+thighPts[1]/2, bone/2)

        if side == 0:
            ptA = np.stack((peak, intersection_rsh[1],intersection_knee[1],intersection_knee[0]))
            self.thighPtsR = ptA
            self.pos0 = pos2D[0]
        else :
            ptA = np.stack((peak, intersection_rsh[0],intersection_knee[0],intersection_knee[1]))
            self.thighPtsL = ptA
            self.pos0 = pos2D[0]
        # Fill up the polygon
        bw_up = ( (A*self.polygonOutline(ptA)))


        ## Calf
        # compute slopes related to the leg position
        slopeKnee = self.findSlope(calfPts[0], calfPts[1])
        a_pen = slopeKnee[0]
        b_pen = slopeKnee[1]
        c_pen = slopeKnee[2]

        # find 2 points corner of the ankle
        intersection_ankle=self.inferedPoint(A,a_pen,b_pen,c_pen,calfPts[0]/2+calfPts[1]/2, bone/2/2)
        if(side!=0): # if two ankles are too close
            if(intersection_ankle[1][0]>(pos2D[14][0]+pos2D[18][0])/2):
                print("two ankles are too close L")
                intersection_ankle[1][0] = (pos2D[14][0]+pos2D[18][0])/2
        else:
            if(intersection_ankle[0][0]<(pos2D[14][0]+pos2D[18][0])/2):
                intersection_ankle[0][0] = (pos2D[14][0]+pos2D[18][0])/2
                print("two ankles are too close R")

        ptA = np.stack((intersection_ankle[1],intersection_ankle[0],intersection_knee[0],intersection_knee[1]))
        if side == 0 :
            self.calfPtsR = ptA
        else:
            self.calfPtsL = ptA
        # Fill up the polygon
        bw_down = (A*self.polygonOutline(ptA))
        return np.array([bw_up,bw_down])

    def legSeg(self,A,side):
        """
        Segment the leg into two body parts
        :param A: depthImag
        :param side: if side = 0 the segmentation will be done for the right leg
                  otherwise it will be for the left leg
        :return: an array containing two body parts : an upper leg and a lower leg
        """

        pos2D = self.pos2D.astype(np.float64)-1

        # Right
        if side == 0 :
            knee =17
            hip = 16
            ankle = 18
        else : # Left
            knee =13
            hip = 12
            ankle = 14

        #check which knee is higher
        if pos2D[17,1] > pos2D[13,1]:
            P = pos2D[17,1]
        else:
            P = pos2D[13,1]
        ## Find the Thigh
        # find the fifth point that can not be deduce simply with Slopes or intersection using the entire hip
        peak1 = self.nearestPeak(A,pos2D[12],pos2D[16],P, pos2D[0])
        if(sum(peak1) == 0): # cannot find the peak
            print "cannot find the peak between legs"
            peak1 = pos2D[0]

        # compute slopes related to the leg position
        slopeThigh = self.findSlope(pos2D[hip],pos2D[knee])
        slopeCalf = self.findSlope(pos2D[ankle],pos2D[knee])
        a_pen = slopeThigh[0] + slopeCalf[0]
        b_pen = slopeThigh[1] + slopeCalf[1]
        if a_pen*b_pen==0:
            a_pen = slopeThigh[1]
            b_pen =-slopeThigh[0]
        c_pen = -(a_pen*pos2D[knee,0]+b_pen*pos2D[knee,1])

        # find lenght of leg
        bone1 = LA.norm(pos2D[knee]-pos2D[ankle])
        bone2 = LA.norm(pos2D[knee]-pos2D[hip])
        bone = max(bone1,bone2)

        # find 2 points corner of the knee
        intersection_knee=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[knee], bone/1.5/2)
        if(side!=0): # if two knees are too close
            if(intersection_knee[1][0]>(pos2D[13][0]+pos2D[17][0])/2):
                print("two knees are too close L")
                intersection_knee[1][0] = (pos2D[13][0]+pos2D[17][0])/2
        else:
            if(intersection_knee[0][0]<(pos2D[13][0]+pos2D[17][0])/2):
                intersection_knee[0][0] = (pos2D[13][0]+pos2D[17][0])/2
                print("two knees are too close R")

        # find right side of the hip rsh
        region = np.zeros(A.shape)
        if side==0:
            region[int(pos2D[1][1]):int(pos2D[hip][1]), int(pos2D[1][0]):-1] = 1
        else:
            region[int(pos2D[1][1]):int(pos2D[hip][1]), 0:int(pos2D[1][0])] = 1
        B1 = np.logical_and( (A==0),region)
        f = np.nonzero(B1)
        if(sum(sum(f))!=0):
            # find the minimum in distance to spine
            d = np.argmin(np.sum( np.square(np.array([pos2D[hip,0]-f[1], pos2D[hip,1]-f[0]]).transpose()),axis=1 ))
            intersection_rsh = np.array([f[1][d],f[0][d]])
        else:
            print "there is no hole beside thespine"
            exit()

        if side == 0:
            #ptA = np.stack((pos2D[0],intersection_rsh,intersection_knee[1],intersection_knee[0],peak1))
            ptA = np.stack((peak1, intersection_rsh,intersection_knee[1],intersection_knee[0]))
            self.thighPtsR = ptA
            self.pos0 = pos2D[0]
        else :
            #ptA = np.stack((pos2D[0],intersection_rsh,intersection_knee[0],intersection_knee[1],peak1))
            ptA = np.stack((peak1, intersection_rsh,intersection_knee[0],intersection_knee[1]))
            self.thighPtsL = ptA
            self.pos0 = pos2D[0]
        # Fill up the polygon
        bw_up = ( (A*self.polygonOutline(ptA)))
        ## Find Calf
        # Define slopes
        a_pen = slopeCalf[1]
        b_pen = -slopeCalf[0]
        c_pen = -(a_pen*pos2D[ankle,0]+b_pen*pos2D[ankle,1])

        # find 2 points corner of the ankle
        intersection_ankle=self.inferedPoint(A,a_pen,b_pen,c_pen,pos2D[ankle], bone/2/2)
        if(side!=0): # if two ankles are too close
            if(intersection_ankle[1][0]>(pos2D[14][0]+pos2D[18][0])/2):
                print("two ankles are too close L")
                intersection_ankle[1][0] = (pos2D[14][0]+pos2D[18][0])/2
        else:
            if(intersection_ankle[0][0]<(pos2D[14][0]+pos2D[18][0])/2):
                intersection_ankle[0][0] = (pos2D[14][0]+pos2D[18][0])/2
                print("two ankles are too close R")

        ptA = np.stack((intersection_ankle[1],intersection_ankle[0],intersection_knee[0],intersection_knee[1]))
        if side == 0 :
            self.calfPtsR = ptA
        else:
            self.calfPtsL = ptA
        # Fill up the polygon
        bw_down = (A*self.polygonOutline(ptA))
        return np.array([bw_up,bw_down])

    def headSeg(self,A):
        """
        Segment the head
        :param A: binary depthImag
        :return: head body part
        """

        pos2D = self.pos2D.astype(np.float64)-1

        #compute slopes Shoulder Head (SH)spine
        slopesSH=self.findSlope(self.peakshoulderL,self.peakshoulderR)
        a_pen = slopesSH[0]
        b_pen = slopesSH[1]
        c_pen = -(a_pen*self.peakshoulderL[0]+b_pen*self.peakshoulderL[1])

        # find left
        x = pos2D[4,0]
        y =int(np.round(-(a_pen*x+c_pen)/b_pen))
        headLeft = np.array([x,y])

        # find right
        x = pos2D[8,0]
        y =int(np.round(-(a_pen*x+c_pen)/b_pen))
        headRight = np.array([x,y])

        # distance head - neck
        h = 2*(pos2D[2,1]-pos2D[3,1])
        #h = 2*(pos2D[2,1]-pos2D[3,1])*2 # MIT
        # create point that higher than the head
        headUp_right = np.array([pos2D[8,0],pos2D[2,1]-h])
        headUp_left = np.array([pos2D[4,0],pos2D[2,1]-h])
        # stock corner of the polyogne
        pt4D = np.array([headUp_right,headUp_left,headLeft,headRight])
        self.headPts = pt4D
        pt4D_bis = np.array([headUp_left,headLeft,headRight,headUp_right])
        # Compute slope of each line of the polygon
        HeadSlope=self.findSlope(pt4D.transpose(),pt4D_bis.transpose())
        # reference point
        midpoint = [pos2D[3,0], pos2D[3,1]]
        ref= np.array([HeadSlope[0]*midpoint[0] + HeadSlope[1]*midpoint[1] + HeadSlope[2]]).astype(np.float32)
        # fill up the polygon
        bw_head = ( A*self.polygon_optimize(HeadSlope,ref,HeadSlope.shape[1]))
        return bw_head

    def GetBody(self,binaryImage):
        """
        Delete all the unwanted connected component from the binary image
        It focuses on the group having the right pos2D, for now the body
        :param binaryImage: binary image of the body but all body part are substracted to the body leaving only the trunck and noise
        :return: trunk
        """

        pos2D = self.pos2D-1
        # find all connected component and label it
        labeled, n = spm.label(binaryImage)
        # Get the labelled  of the connected component that have the trunk
        threshold = labeled[pos2D[1,1],pos2D[1,0]]
        # erase all connected component that are not the trunk
        labeled = (labeled==threshold)
        return labeled

    def GetHand(self,binaryImage,side, re=0):
        """
        Delete all the little group unwanted from the binary image
        It focuses on the group having the right pos2D, here the hands
        :param binaryImage: binary image of the body without limbs
        :param side: if side = 0 the segmentation will be done for the right hand
                  otherwise it will be for the left hand
        :param re: is resegment or not
        :return: one hand
        """
        # Right side
        if side == 0 :
            idx = 11 #hand
            wrist = 10 #wrist
            handtip = 23 #hand tip
            handtip1 = 21
            thumb = 24
            idx1 = 7 # the other hand
        # Left side
        else :
            idx =7
            wrist = 6
            handtip = 21
            handtip1 = 23
            thumb = 22
            idx1 = 11
        pos2D = self.pos2D-1

        #create a sphere of radius 12 so that anything superior does not come in the feet label
        handDist = 25#
        #handDist = 50 #MIT
        #handDist = (max(LA.norm( (pos2D[wrist]-pos2D[idx])), LA.norm( (pos2D[handtip]-pos2D[idx])))*1.5).astype(np.int16)
        #since feet are on the same detph as the floor some processing are required before using cc
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        mask = np.ones([line,col,2])
        mask = mask*pos2D[idx]
        #create a matrix containing in each pixel its index
        lineIdx = np.array([np.arange(line) for _ in range(col)]).transpose()
        colIdx = np.array([np.arange(col) for _ in range(line)])
        ind = np.stack( (colIdx,lineIdx), axis = 2)
        #compute the distance between the skeleton point of feet and each pixel
        mask = np.sqrt(np.sum( (ind-mask)*(ind-mask),axis = 2))
        mask = (mask < handDist)
        # compute the center between two foot
        centerpos = (pos2D[idx,0]+pos2D[idx1,0])*1.0/2
        if(abs(pos2D[idx,0]-pos2D[idx1,0])<0.001):
            print "junctions of two hand are in the same x axis"
            if side==0:
                pos2D[idx,0]=pos2D[idx,0]+0.1
                pos2D[idx1,0]=pos2D[idx1,0]-0.1
            else:
                pos2D[idx,0]=pos2D[idx,0]-0.1
                pos2D[idx1,0]=pos2D[idx1,0]+0.1
        mask2 = np.ones([line,col])*centerpos
        if(pos2D[idx,0]-centerpos>=0):
            mask2 = (ind[:,:,0]-mask2)>=0
        else:
            mask2 = (ind[:,:,0]-mask2)<=0

        mask = mask * binaryImage * mask2

        # compute the body part as it is done for the head
        labeled, n = spm.label(mask)
        threshold = labeled[pos2D[idx,1],pos2D[idx,0]]
        if re==1:
            threshold = 0
            for i in range(n+1):
                if sum(sum(((labeled==threshold)*mask))) <sum(sum(((labeled==i)*mask))):
                    threshold = i
        elif(binaryImage[pos2D[idx,1],pos2D[idx,0]]==0 ): # meet hole and noise in depth image
            print("meet hand's hole")
            if(binaryImage[pos2D[idx,1],pos2D[idx,0]+2]!=0):
                print("hand right")
                threshold =  labeled[pos2D[idx,1],pos2D[idx,0]+2]
            elif(binaryImage[pos2D[idx,1],pos2D[idx,0]-2]!=0):
                print("hand left")
                threshold =  labeled[pos2D[idx,1],pos2D[idx,0]-2]
            elif(binaryImage[pos2D[idx,1]+1,pos2D[idx,0]]!=0):
                print("hand lower")
                threshold =  labeled[pos2D[idx,1]+1,pos2D[idx,0]]
            elif(binaryImage[pos2D[idx,1]-1,pos2D[idx,0]]!=0):
                print("hand upper")
                threshold =  labeled[pos2D[idx,1]-1,pos2D[idx,0]]
            elif(binaryImage[pos2D[wrist,1]+1,pos2D[wrist,0]]!=0 and (pos2D[wrist,0]-centerpos)*(pos2D[idx,0]-centerpos)>=0 ):
                print("wrist lower")
                threshold =  labeled[pos2D[wrist,1]+1,pos2D[wrist,0]]
            elif(binaryImage[pos2D[wrist,1]-1,pos2D[wrist,0]]!=0 and (pos2D[wrist,0]-centerpos)*(pos2D[idx,0]-centerpos)>=0 ):
                print("wrist upper")
                threshold =  labeled[pos2D[wrist,1]-1,pos2D[wrist,0]]
            elif(binaryImage[pos2D[handtip,1],pos2D[handtip,0]]!=0 and (pos2D[handtip,0]-centerpos)*(pos2D[idx,0]-centerpos)>=0 ):
                print("handtip")
                threshold =  labeled[pos2D[handtip,1],pos2D[handtip,0]]
            elif(binaryImage[pos2D[thumb,1],pos2D[thumb,0]]!=0 and (pos2D[thumb,0]-centerpos)*(pos2D[idx,0]-centerpos)>=0 ):
                print("thumb")
                threshold =  labeled[pos2D[thumb,1],pos2D[thumb,0]]
            elif(binaryImage[pos2D[handtip1,1],pos2D[handtip1,0]]!=0 and (pos2D[handtip1,0]-centerpos)*(pos2D[idx,0]-centerpos)>=0 ):
                print("handtip wrong")
                threshold =  labeled[pos2D[handtip1,1],pos2D[handtip1,0]]
            else:
                threshold = 1
                if side==0:
                    print("cannot find the R hand")
                else:
                    print("cannot find the L hand")

        labeled = (labeled==threshold)
        return labeled


    def GetFoot(self,binaryImage,side, re=0):
        """
        Delete all the little group unwanted from the binary image
        It focuses on the group having the right pos2D, here the feet
        :param binaryImage: binary image of the body without limbs
        :param side: if side = 0 the segmentation will be done for the right feet
                  otherwise it will be for the left feet
        :param re: is resegment or not
        :return: one feet
        """


        #Right Side
        if side == 0 :
            idx =19
            disidx = 18 #ankle
            idx1 = 15 #foot of another side
        # Left Side
        else :
            idx =15
            disidx = 14
            idx1 = 19
        pos2D = self.pos2D-1

        #create a sphere mask1 of radius 12 so that anything superior does not come in the feet label
        footDist = 25
        #footDist = 50
        #footDist = (LA.norm((pos2D[disidx]-pos2D[idx]))*1.5).astype(np.int16)
        #since feet are on the same detph as the floor some processing are required before using cc
        line = self.depthImage.shape[0]
        col = self.depthImage.shape[1]
        mask1 = np.ones([line,col,2])
        mask1 = mask1*pos2D[idx]
        #create a matrix containing in each pixel its index
        lineIdx = np.array([np.arange(line) for _ in range(col)]).transpose()
        colIdx = np.array([np.arange(col) for _ in range(line)])
        ind = np.stack( (colIdx,lineIdx), axis = 2)
        #compute the distance between the skeleton point of feet and each pixel
        mask1 = np.sqrt(np.sum( (ind-mask1)*(ind-mask1),axis = 2))
        mask1 = (mask1 < footDist) # distance < radius
        # compute the center between two foot
        centerpos = (pos2D[idx,0]+pos2D[idx1,0])/2
        mask2 = np.ones([line,col])*centerpos
        if(pos2D[idx,0]-centerpos>=0):
            mask2 = (ind[:,:,0]-mask2)>=0
        else:
            mask2 = (ind[:,:,0]-mask2)<=0
        mask = mask1 * binaryImage *mask2
        # compute the body part as it is done for the head
        labeled, n = spm.label(mask)
        threshold = labeled[pos2D[idx,1],pos2D[idx,0]]
        if re==1:
            threshold = 0
            for i in range(n+1):
                if sum(sum(((labeled==threshold)*mask))) < sum(sum(((labeled==i)*mask))):
                    threshold = i
        elif(binaryImage[pos2D[idx,1],pos2D[idx,0]]==0): # meet hole and noise in depth image
            print("meet foot's hole")
            #exit()
            if(binaryImage[pos2D[disidx,1]+2,pos2D[disidx,0]]!=0):
                print("ankle lower")
                threshold = labeled[pos2D[disidx,1]+2,pos2D[disidx,0]]
            elif(binaryImage[pos2D[disidx,1]-1,pos2D[disidx,0]]!=0):
                print("ankle upper")
                threshold = labeled[pos2D[disidx,1]-1,pos2D[disidx,0]]
            elif(binaryImage[pos2D[idx,1],pos2D[idx,0]+1]!=0):
                print("foot right")
                threshold = labeled[pos2D[idx,1],pos2D[idx,0]+1]
            elif(binaryImage[pos2D[idx,1],pos2D[idx,0]-1]!=0):
                print("foot left")
                threshold = labeled[pos2D[idx,1],pos2D[idx,0]-1]
            else:
                print("cannot find the Foot")
                threshold = 0
                for i in range(n):
                    if sum(sum(((labeled==threshold)*mask))) <sum(sum(((labeled==i)*mask))):
                        threshold = i
                if threshold==0:
                    threshold = 1000

        labeled = (labeled==threshold)
        return labeled


















