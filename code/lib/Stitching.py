# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:50:19 2017

@author: Inoe ANDRE
"""

import numpy as np
import math
from math import cos, sin
from numpy import linalg as LA
import imp
PI = math.pi

General = imp.load_source('General', './lib/General.py')

class Stitch():
    """
    All method that concern linking or aligning body parts together are in Stitch
    """
    def __init__(self, number_bodyPart):
        """
        Constructor
        :param number_bodyPart: number of body parts
        """
        self.nb_bp = number_bodyPart
        self.StitchedVertices = 0
        self.StitchedFaces = 0

    def NaiveStitch(self, PartVtx,PartNmls,PartFaces,PoseBP):
        """
        Add the vertices and faces of each body parts
        together after transforming them in the global coordinates system
        :param PartVtx: List of vertices for a body parts
        :param PartNmls: List of normales for a body parts
        :param PartFaces:  List of faces for a body parts
        :param PoseBP: local to global transform
        :return: none
        """
        #Initialize values from the list of
        ConcatVtx = self.StitchedVertices
        ConcatFaces = self.StitchedFaces
        ConcatNmls = self.StitchedNormales

        # tranform the vertices in the global coordinates system
        PartVertices = self.TransformVtx(PartVtx,PoseBP,1)
        PartNormales = self.TransformNmls(PartNmls,PoseBP,1)
        PartFacets = PartFaces + np.max(ConcatFaces)+1

        # concatenation
        self.StitchedVertices = np.concatenate((ConcatVtx,PartVertices))
        self.StitchedNormales = np.concatenate((ConcatNmls,PartNormales))
        self.StitchedFaces = np.concatenate((ConcatFaces,PartFacets))
        

        
    def TransformVtx(self, Vtx,Pose, s):
        """
        Transform the vertices in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        :param Vtx: List of vertices
        :param Pose:  local to global transform
        :param s: subsampling factor
        :return: list of transformed vertices
        """
        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        Vtx = np.dot(pt,Pose.T)
        return Vtx[:,0:3]
        
    def TransformNmls(self, Nmls,Pose, s):
        """
        Transform the normales in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        :param Nmls:  List of normales
        :param Pose: local to global transform
        :param s: subsampling factor
        :return: list of transformed normales
        """
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Pose[0:3,0:3].T)
        return nmle

    def RArmsTransform(self, angle,bp, pos2d,RGBD,Tg):
        """
        Transform Pose matrix to move the model of the right arm
        For now just a rotation in the z axis
        :param bp : number of the body parts
        :param pos2d : position in 2D of the junctions
        :param RGBD : an RGBD object containing the image
        :param Tg : local to global transform
        TEST FUNCTION : TURN THE LEFT ARM OF THE SEGMENTED BODY.
        """

        # Rotate skeleton right arm
        angley = angle  # pi * 2. * delta_x / float(Size[0])
        RotZ = np.array([[cos(angley), -sin(angley), 0.0, 0.0], \
                         [sin(angley), cos(angley), 0.0, 0.0], \
                         [0.0, 0.0, 1.0, 0.0], \
                         [0.0, 0.0, 0.0, 1.0]], np.float32)

        ctr = pos2d[4].astype(np.int)
        rotAxe = Tg[7][0:3, 3]
        ctr3D = Tg[bp][0:3, 3]

        # # Rotation of about 30
        if bp == 1 :
            # # transform joints
            print pos2d[5:8]
            rotat = RotZ[0:2, 0:2]
            for rot in range(3):
                pt = (pos2d[5 + rot]).astype(np.int)- ctr
                pt = np.dot(rotat[0:2, 0:2], pt.T).T
                pos2d[5+rot] = pt + ctr

        if bp ==1 or bp == 2 or bp==12:
            if bp == 12:
                #pos = 4 # should left
                pos = 7  # hand left
                Xm = pos2d[pos,0]
                Ym = pos2d[pos,1]
            elif bp == 2:
                pos = 5 #elbow left
                Xm = (pos2d[pos,0] + pos2d[pos-1,0])/2
                Ym = (pos2d[pos,1] + pos2d[pos-1,1])/2
            elif bp == 1:
                pos = 6  # wrist left
                Xm = (pos2d[pos,0] + pos2d[pos-1,0])/2
                Ym = (pos2d[pos,1] + pos2d[pos-1,1])/2

            ctr = Tg[bp][0:3,3]

            # for rot in range(3):
            #     pt = (pos2d[5 + rot]).astype(np.int)- ctr
            #     pt = np.dot(rotat[0:2, 0:2], pt.T).T
            #     pos2d[5+rot] = pt + ctr

            print ctr
            print ctr
            print RGBD.Vtx[pos2d[pos,0],pos2d[pos,1]]
            print RGBD.Vtx[pos2d[pos,0],pos2d[pos,1]]
            z = ctr[2]

            ctr[0] = z * (Xm - RGBD.intrinsic[0, 2]) / RGBD.intrinsic[0, 0]
            ctr[1] = z * (Ym - RGBD.intrinsic[1, 2]) / RGBD.intrinsic[1, 1]
            print ctr
            print ctr
            RotZ[0:3, 3] = ctr
            Tg[bp][:,0:3] = np.dot(RotZ, Tg[bp][:,0:3])
            #Tg[bp][0:3, 3] = ctr
            print RotZ



    def GetBBTransfo(self, pos2d,cur,prev,RGBD ,pRGBD, nRGBD, bp, pose):
        """
        Transform Pose matrix to move the model body parts according to the position of the skeleton
        For now just a rotation in the z axis
        :param bp : number of the body parts
        :param pos2d : position in 2D of the junctions
        :param cur : index for the current frame
        :param prev : index for the previous frame
        :param RGBD : an RGBD object containing the image
        :param pRGBD : an RGBD object containing the vertex in pre frame
        :param nRGBD : an RGBD object containing the vertex in now frame
        :param pose : the camera transformation from prev to cur
        :return The transform between two skeleton
        """
        PosCur = pos2d[0,cur]
        PosPrev = pos2d[0,prev]

        # print prev
        # print cur
        # get the junctions of the current body parts
        pos = self.GetPos(bp)

        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)

        # Compute the transform between the two skeleton : Tbb = A^-1 * B

        # Compute A
        A = self.GetCoordSyst(PosPrev,pos,RGBD, pRGBD, bp, pose)
        # Compute B
        B = self.GetCoordSyst(PosCur, pos, RGBD, nRGBD, bp, Id4)
        # check if junction is on the noise
        if B[3,3]==0:
            print bp, " meet noise"

            return pose

        # Compute Tbb : skeleton tracking transfo
        Tbb = np.dot(B,General.InvPose(A))#B#np.identity(4)#A#

        # get vector a,b
        a = np.array([A[0,0], A[1,0], A[2,0]])
        b = np.array([B[0,0], B[1,0], B[2,0]])

        # calculate Rotation from a to b
        R = self.GetRotatefrom2Vector(a,b)
        R_T = np.identity(4)
        R_T[:3, :3] = R
        T_T = np.identity(4)
        T_T[0:3,3] = -A[0:3,3]
        result = np.dot(R_T, T_T)
        T_T[0:3,3] = B[0:3,3]
        result = np.dot(T_T, result)

        # print A
        # print Tg
        # print B
        # print "TBB"
        # print result
        # print Tbb

        return result
    
    def GetRotatefrom2Vector(self, a, b):
        '''
        calculate the Rotation matrix form vector a to vector b
        :param a: start vector
        :param b: end vector
        :return: Rotation Matrix
        '''

        x = np.cross(a,b)/LA.norm(np.cross(a,b))
        theta = np.arccos(np.dot(a,b)/np.dot(LA.norm(a),LA.norm(b)))
        Ax = np.array([[0., -x[2], -x[1]], [x[2], 0., -x[0]], [-x[1], x[0], 0.]])
        R = np.identity(3)
        R += np.sin(theta)*Ax + (1-np.cos(theta))*np.dot(Ax,Ax)
        
        if(theta*180<5):
            print "angle too small"
        if(180-theta*180<5):
            print "angle closes 180"
        
        return R


    def GetCoordSyst(self, pos2d,jt,RGBD, vRGBD, bp, pose):
        '''
        This function compute the coordinates system of a body part according to the camera pose
        :param pos2d: position in 2D of the junctions
        :param jt: junctions of the body parts
        :param RGBD: Image
        :param vRGBD: vertex in pref
        :param bp: number of body part
        :param pose : the camera transformation from prev to cur
        :return: Matrix containing the coordinates systems
        '''
        # compute the 3D centers point of the bounding boxes using the skeleton
        ctr = np.array([0.0, 0.0, 0.0, 1.0], np.float)
        Tg = RGBD.TransfoBB[bp]
        Tg = np.dot(pose, Tg)
        z = Tg[2,3]
        pos2d = pos2d.astype(np.int16)-1
        if bp < 9 or bp == 12:
            ctr[0] = (pos2d[jt[0], 0] + pos2d[jt[1], 0]) / 2
            ctr[1] = (pos2d[jt[0], 1] + pos2d[jt[1], 1]) / 2
            #ctr[2] = z
            ctr[2] = (vRGBD.depth_image[pos2d[jt[0], 1],pos2d[jt[0], 0]] + vRGBD.depth_image[pos2d[jt[1], 1],pos2d[jt[1], 0]]) / 2
            if(vRGBD.depth_image[pos2d[jt[0], 1],pos2d[jt[0], 0]]==0 or vRGBD.depth_image[pos2d[jt[1], 1],pos2d[jt[1], 0]]==0):
                print bp, " meet noise at center"
                ctr[2] = z
        else:
            ctr[0] = pos2d[jt[2], 0]
            ctr[1] = pos2d[jt[2], 1]
            #ctr[2] = z
            ctr[2] = vRGBD.depth_image[pos2d[jt[2], 1],pos2d[jt[2], 0]]
            if(ctr[2]==0):
                print bp, " meet noise at center"
                ctr[2] = z

        # compute the center of the coordinates system
        ctr[0] = ctr[2] * (ctr[0]-vRGBD.intrinsic[0,2])/vRGBD.intrinsic[0, 0]
        ctr[1] = ctr[2] * (ctr[1]-vRGBD.intrinsic[1,2])/vRGBD.intrinsic[1, 1]
        ctr = np.dot(ctr, pose.T)
        ctr = ctr[0:3]

        # Compute first junction points  of current frame
        pt1 = np.array([0.0, 0.0, 0.0, 1.0], np.float)
        pt1[0] = (pos2d[jt[1], 0]-vRGBD.intrinsic[0,2])/vRGBD.intrinsic[0, 0]
        pt1[1] = (pos2d[jt[1], 1]-vRGBD.intrinsic[1,2])/vRGBD.intrinsic[1, 1]
        pt1[2] = vRGBD.depth_image[pos2d[jt[1], 1],pos2d[jt[1], 0]]
        if(pt1[2]==0):
            print bp, " meet noise at pt1"
            pt1[2] = z
        pt1[0] *= pt1[2]
        pt1[1] *= pt1[2]
        # Compute second junction points  of current frame
        pt2 = np.array([0.0, 0.0, 0.0, 1.0], np.float)
        pt2[0] = (pos2d[jt[0], 0]-vRGBD.intrinsic[0,2])/vRGBD.intrinsic[0, 0]
        pt2[1] = (pos2d[jt[0], 1]-vRGBD.intrinsic[1,2])/vRGBD.intrinsic[1, 1]
        pt2[2] = vRGBD.depth_image[pos2d[jt[0], 1],pos2d[jt[0], 0]]
        if(pt2[2]==0):
            print bp, " meet noise at pt2"
            pt2[2] = z
        pt2[0] *= pt2[2]
        pt2[1] *= pt2[2]
        
        # do camera transformation
        pt1 = np.dot(pt1, pose.T)
        pt1 = pt1[0:3]
        pt2 = np.dot(pt2, pose.T)
        pt2 = pt2[0:3]

        # Compute normalized axis of coordinates system
        axeX = (pt2 - pt1)/LA.norm(pt2 - pt1)
        #signX = np.sign(axeX)
        #axeX = signX[1]*axeX
        axeZ = np.array([0.0, 0.0, 1.0], np.float)
        axeY = General.normalized_cross_prod(axeX, axeZ)
        if bp == 12 :
            axeX = (pt1 - pt2) / LA.norm(pt1 - pt2)
            axeZ = np.array([0.0, 0.0, 1.0], np.float)
            axeY = General.normalized_cross_prod(axeX, axeZ)

        # Bounding boxes tracking matrix
        e1b = np.array( [axeX[0],axeX[1],axeX[2],0])
        e2b = np.array( [axeY[0],axeY[1],axeY[2],0])
        e3b = np.array( [axeZ[0],axeZ[1],axeZ[2],0])
        origine = np.array( [ctr[0],ctr[1],ctr[2],1])
        coord = np.stack( (e1b,e2b,e3b,origine),axis = 0 ).T

        # check if the junction has depth value
        if pt1[2] * pt2[2]==0:
            coord[3,3] = 0

        return coord

    def GetPos(self,bp):
        '''
        According to the body parts, get the correct index of junctions
        mid is used to get the center while pos1 and pos2 give extremes junctions of the body parts
        :param bp: number of the body part
        :return: return the junctions corresponding to the body parts
        '''
        mid = 0
        if bp ==1 :
            pos1 = 6  # wrist left
            pos2 = 5 # elbow left
        elif bp == 2:
            pos1 = 4  # elbow left
            pos2 =  5#  shoulder left
        elif bp == 3:
            pos1 = 10 # wrist right
            pos2 =  9 # elbow left
        elif bp == 4:
            pos1 = 9 # elbow left
            pos2 =  8 # shoulder left
        elif bp == 5:
            pos1 = 17  #knee right
            pos2 = 16 #hip right
        elif bp == 6:
            pos1 = 18  # ankle right
            pos2 = 17 # knee right
        elif bp == 7:
            pos1 = 13 # knee left
            pos2 = 12 # hip left
        elif bp == 8:
            pos1 = 14 # ankle left
            pos2 =  13 # knee left
        elif bp == 9:
            pos1 = 3 # head
            pos2 = 2 # neck
            mid = 3
        elif bp == 10:
            pos1 = 0  # spine base
            pos2 = 20 # spine should
            mid = 1 # spine mid
        elif bp == 11:
            pos1 = 11  # hand right
            pos2 = 10 # wrist right
            mid = 11
        elif bp == 12:
            pos1 = 7  # hand left
            pos2 = 6  # wrist left
            #mid == 7
        elif bp == 13:
            pos1 = 15  # foot left
            pos2 = 14 # ankle left
            mid = 15
        elif bp == 14:
            pos1 = 19 # foot right
            pos2 = 18  # ankle right
            mid = 19

        return np.array([pos1,pos2,mid])