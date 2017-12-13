# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 17:50:19 2017

@author: Inoe ANDRE
"""

import cv2
import numpy as np
import math
from math import cos, sin
from numpy import linalg as LA
import imp    
from skimage.draw import line_aa
import copy
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
        self.StitchedNormales = 0

    def NaiveStitch(self, PartVtx,PartNmls,PartFaces, Tg, coordGbl_pre=np.zeros(0), BBTrans=np.zeros(0)):
        """
        Add the vertices and faces of each body parts
        together after transforming them in the global coordinates system
        :param PartVtx: List of vertices for a body parts
        :param PartNmls: List of normales for a body parts
        :param PartFaces:  List of faces for a body parts
        :param Tg:  local to global transform
        :param coordsGbl_pre: the corners of bounding-box in global coord
        :param BBTrans: the transform matrix of each corners in global bounding-box
        :return: none
        """
        #Initialize values from the list of
        ConcatVtx = self.StitchedVertices
        ConcatFaces = self.StitchedFaces
        ConcatNmls = self.StitchedNormales

        # tranform the vertices in the global coordinates system
        PartVertices = self.TransformVtx(PartVtx,Tg, coordGbl_pre, BBTrans, 1)
        PartNormales = self.TransformNmls(PartNmls,Tg,1)
        PartFacets = PartFaces + np.max(ConcatFaces)+1

        # concatenation
        self.StitchedVertices = np.concatenate((ConcatVtx,PartVertices))
        self.StitchedNormales = np.concatenate((ConcatNmls,PartNormales))
        self.StitchedFaces = np.concatenate((ConcatFaces,PartFacets))
        

        
    def TransformVtx(self, Vtx, Tg, coordGbl=np.zeros(0), BBTrans=np.zeros(0), s=1):
        """
        Transform the vertices in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        :param Vtx: List of vertices
        :param Tg:  local to global transform
        :param coordsGbl: the corners of bounding-box in global coord
        :param coordsGblTrans: the transform matrix of each corners in global bounding-box
        :param s: subsampling factor
        :return: list of transformed vertices
        """
        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        pt = np.dot(pt, Tg.T)
        Vtx = pt[:,0:3]
        
        if coordGbl.shape[0]==0:
            return Vtx

        newVtx = np.zeros((Vtx.shape[0], 4))
        weights = np.zeros((Vtx.shape[0], coordGbl.shape[0]), dtype=np.double)

        for i in range(coordGbl.shape[0]):
            weights[:, i] = 1/LA.norm(Vtx-coordGbl[i,:], axis=1)
            Pose = BBTrans[i]
            newVtx[:,0] += np.dot(pt,Pose.T)[:,0]*weights[:,i]
            newVtx[:,1] += np.dot(pt,Pose.T)[:,1]*weights[:,i]
            newVtx[:,2] += np.dot(pt,Pose.T)[:,2]*weights[:,i]
            newVtx[:,3] += np.dot(pt,Pose.T)[:,3]*weights[:,i]

        newVtx[:,0] /= np.sum(weights, axis=1)
        newVtx[:,1] /= np.sum(weights, axis=1)
        newVtx[:,2] /= np.sum(weights, axis=1)
        newVtx[:,3] /= np.sum(weights, axis=1)

        return newVtx[:,0:3]
        
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

    def TransfoBBcorners(self, prevPos, BB, BBTrans):
        """
        Transform the corners of bounding-boxes according to the position of the skeleton
        :param prevPos: position in 3D of the junctions
        :param BB: the global bounding-boxes in previous frame
        :param BBTrans: the transform matrix of  bounding-boxes in previous frame
        :return: the new bounding-boxes in global and the transform matrix of bounding-boxes
        """
        '''
        AllrelatedJun = [ [],\
        [[5,4,6],[5,4,6],[6,5,7],[6,5,7]], [[5,4,6],[4,5,20,1],[4,5,20,1],[5,4,6]], \
        [[9,8,10],[9,8,10],[10,11,9],[10,11,9]], [[9,8,10],[9,8,10],[9,8,20,1],[9,8,20,1]], \
        [[16,0,17,1,12,13],[16,17,0,1],[17,16,18],[17,16,18]], [[18,17,19],[18,17,19],[17,16,18],[17,16,18]], \
        [[12,0,13,1,16,17],[13,12,14],[13,12,14],[12,0,13,1]], [[14,13,15],[14,13,15],[13,12,14],[13,12,14]], \
        [[2,20,4,1],[3,2],[3,2],[2,20,8,1]], \
        [[4,5,20,1],[4,5,20,1],[2,20,4,1],[2,20,8,1],[8,9,20,1],[8,9,20,1],[16,17,0,1],[16,12,0,1,17,13],[12,0,13,1]], \
        [[10,9,11],[10,9,11],[10,11],[10,11]], [[6,5,7],[6,5,7],[6,7],[6,7]], \
        [[14,13,15],[14,13,15],[14,15],[14,15]], [[18,17,19],[18,17,19],[18,19],[18,19]]
        ]
        '''
        AllrelatedBone = [ [],\
        [[0,1],[0,1],[0],[0]], [[0,1],[1,2],[1,2],[0,1]], \
        [[3,4],[3,4],[3],[3]], [[3,4],[3,4],[4,5],[4,5]], \
        [[11,8,15],[11,10],[10,9],[10,9]], [[9],[9],[9,10],[9,10]], \
        [[8,11,15],[7,6],[7,6],[7,8]], [[6],[6],[6,7],[6,7]], \
        [[2,14,12],[13,12],[13,12],[5,12,14]], \
        [[1,2],[1,2],[2,12,14],[5,12,14],[5,4],[5,4],[11,10],[11,8,15],[7,8]], \
        [[3],[3],[3],[3]], [[0],[0],[0],[0]], \
        [[6],[6],[6],[6]], [[9],[9],[9],[9]]
        ]
        #'''
        bonelist = [[5,6],[4,5],[20,4],[9,10],[8,9],[20,8], \
        [13,14],[12,13],[0,12],[17,18],[16,17],[0,16], \
        [20,2],[2,3],[20,1],[0,1]]

        newBBs=[]
        newBBs.append(np.array((0,0,0)))
        newBBTrans = []
        newBBTrans.append(np.identity(4))
        for bp in range(1,len(AllrelatedBone)):
            relatedBoneList = AllrelatedBone[bp]
            newBB = np.zeros((len(BB[bp]),3))
            newBBTran = np.zeros((len(BB[bp]),4,4))
            for p in range(len(relatedBoneList)):
                relatedBones = relatedBoneList[p]
                point = BB[bp][p]
                pt = np.array([0.,0.,0.,1.])
                pt[0:3] = BB[bp][p]
                weights = []
                for r in range(len(relatedBones)):
                    relatedBone = relatedBones[r]
                    bonecenter = prevPos[bonelist[relatedBone][0]]/2+prevPos[bonelist[relatedBone][1]]/2
                    weights.append(1/np.linalg.norm(bonecenter-point))
                    newBBTran[p] += weights[r]*self.boneTrans[relatedBone]
                newBBTran[p] /= sum(weights)
                newBB[p,:] = np.dot(newBBTran[p,:,:], pt.T)[0:3]
                newBBTran[p, :, :] = np.dot(newBBTran[p,:,:], BBTrans[bp][p,:,:])
            for p in range(len(relatedBoneList),len(relatedBoneList)*2):
                relatedBones = relatedBoneList[p-len(relatedBoneList)]
                point = BB[bp][p]
                pt = np.array([0.,0.,0.,1.])
                pt[0:3] = BB[bp][p]
                weights = []
                for r in range(len(relatedBones)):
                    relatedBone = relatedBones[r]
                    bonecenter = prevPos[bonelist[relatedBone][0]]/2+prevPos[bonelist[relatedBone][1]]/2
                    weights.append(1/np.linalg.norm(bonecenter-point))
                    newBBTran[p] += weights[r]*self.boneTrans[relatedBone]
                newBBTran[p] /= sum(weights)
                newBB[p,:] = np.dot(newBBTran[p,:,:], pt.T)[0:3]
                newBBTran[p, :, :] = np.dot(newBBTran[p,:,:], BBTrans[bp][p,:,:])
            newBBs.append(newBB)
            newBBTrans.append(newBBTran)
        return newBBs, newBBTrans

    def GetVBonesTrans(self, skeVtx_cur, skeVtx_prev, newRGBD):
        """
        Get transform matrix of bone from previous to current frame
        :param skeVtx_cur: the skeleton Vtx in current frame
        :param skeVtx_prev: the skeleton Vtx in previous frame
        :return: calculated SkeVtx
        """
        bonelist = [[5,6],[4,5],[20,4],[9,10],[8,9],[20,8], \
        [13,14],[12,13],[0,12],[17,18],[16,17],[0,16], \
        [20,2],[2,3],[20,1],[0,1]]

        # test
        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        a1 = newRGBD.depth_image.astype(np.double)/10
        a2 = newRGBD.depth_image.astype(np.double)/10
        a3 = newRGBD.depth_image.astype(np.double)/10

        self.boneTrans = np.zeros((16, 4, 4))
        tempSkeVtx = copy.deepcopy(skeVtx_prev)
        for bIdx in range(len(bonelist)):
            bone = bonelist[bIdx]
            v1 = skeVtx_cur[bone[1]]-skeVtx_cur[bone[0]]
            v2 = skeVtx_prev[bone[1]]-skeVtx_prev[bone[0]]
            T = skeVtx_cur[bone[0]]/2+skeVtx_cur[bone[1]]/2-skeVtx_prev[bone[0]]/2-skeVtx_prev[bone[1]]/2
            R = self.GetRotatefrom2Vector(v2, v1)
            R_T = np.identity(4)
            R_T[0:3,0:3] = R
            T_T = np.identity(4)
            T_T1 = np.identity(4)
            T_T[0:3, 3] = -skeVtx_prev[bone[0]]
            T_T1[0:3, 3] = skeVtx_prev[bone[0]]
            T_T2 = np.identity(4)
            T_T2[0:3, 3] = T
            self.boneTrans[bIdx] = np.identity(4)
            self.boneTrans[bIdx] = np.dot(T_T2, np.dot(T_T1, np.dot(R_T, T_T)))
            
            #update skeleton Vtx
            pt = np.array([0.,0.,0.,1.])
            pt[0:3] = skeVtx_prev[bone[0]]
            if tempSkeVtx[bone[0]][0]!=skeVtx_prev[bone[0]][0]:
                tempSkeVtx[bone[0]] += np.dot(self.boneTrans[bIdx,:,:], pt.T)[0:3]
                tempSkeVtx[bone[0]] /=2
            else:
                tempSkeVtx[bone[0]] = np.dot(self.boneTrans[bIdx,:,:], pt.T)[0:3]
            pt[0:3] = skeVtx_prev[bone[1]]
            if tempSkeVtx[bone[1]][0]!=skeVtx_prev[bone[1]][0]:
                tempSkeVtx[bone[1]] += np.dot(self.boneTrans[bIdx,:,:], pt.T)[0:3]    
                tempSkeVtx[bone[1]] /=2
            else:
                tempSkeVtx[bone[1]] = np.dot(self.boneTrans[bIdx,:,:], pt.T)[0:3]

            # test
            #print 
            '''
            print bIdx
            print skeVtx_cur[bone[0]]
            print skeVtx_cur[bone[1]]
            print skeVtx_prev[bone[0]]
            print skeVtx_prev[bone[1]]
            print v1 
            print v2
            '''
            '''
            #draw
            #origin
            p0 = newRGBD.GetProjPts2D_optimize([skeVtx_cur[bone[0]]], Id4).astype(np.int16)
            p0 = p0[0]
            p1 = newRGBD.GetProjPts2D_optimize([skeVtx_cur[bone[1]]], Id4).astype(np.int16)
            p1 = p1[0]
            rr,cc,val = line_aa(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
            a1[rr,cc] = 1.0
            cv2.imshow("cur",a1)
            #origin pre
            p0 = newRGBD.GetProjPts2D_optimize([skeVtx_prev[bone[0]]], Id4).astype(np.int16)
            p0 = p0[0]
            p1 = newRGBD.GetProjPts2D_optimize([skeVtx_prev[bone[1]]], Id4).astype(np.int16)
            p1 = p1[0]
            rr,cc,val = line_aa(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
            a3[rr,cc] = 1.0
            cv2.imshow("pre",a3)
            #new
            p0 = newRGBD.GetProjPts2D_optimize([tempSkeVtx[bone[0]]], Id4).astype(np.int16)
            p0 = p0[0]
            p1 = newRGBD.GetProjPts2D_optimize([tempSkeVtx[bone[1]]], Id4).astype(np.int16)
            p1 = p1[0]
            rr,cc,val = line_aa(int(p0[1]), int(p0[0]), int(p1[1]), int(p1[0]))
            rr = np.maximum(0,np.minimum(rr, 424-1))
            cc = np.maximum(0,np.minimum(cc, 512-1))
            a2[rr,cc] = 1.0
            cv2.imshow("prev->cur",a2)
        cv2.waitKey(1)
        
        '''
        return tempSkeVtx



    def GetSkeTransfo(self, pos2d,cur,prev,RGBD ,pRGBD, nRGBD, bp, pose):
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
        a = a/LA.norm(a)
        b = b/LA.norm(b)
        v = np.cross(a,b)
        c = np.dot(a,b)
        Av = np.array([[0., -v[2], v[1]],[v[2], 0., -v[0]],[-v[1], v[0], 0.]]) 
        R = np.identity(3)
        R += Av + 1/(1+c)*np.dot(Av,Av)
        
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
    
    def blendMesh(self, pca, bb_cur, bb_prev, vertices):
        '''
        change vertices according bb_prev to bb_cur
        :param pca: RGBD[0].pca[bp]
        :param bb_cur: bounding-boxes in current frame in global coordinate
        :param bb_prev: bounding-boxes in previous frame in global coordinate
        :param vertices: vertices in previous frame in local coordinate
        :return: vertices in current frame in local coordinate
        '''
        '''
        #initial
        bbL_cur = pca.transform(bb_cur)
        bbL_prev = pca.transform(bb_prev)
        vertices_new = np.zeros((vertices.shape[0], 3))
        #blending
        for i in range(vertices.shape[0]):
            weights = []
            for b in range(bbL_cur.shape[0]):
                weights.append(1/np.linalg.norm(vertices[i,:]-bbL_prev[b,:]))
                vertices_new[i,:]+=(bbL_cur[b,:]-bbL_prev[b,:])*weights[b]
            vertices_new[i,:]/=sum(weights)
            vertices_new[i,:]+=vertices[i,:]

        return vertices_new
        '''
        #initial
        bbL_cur = pca.transform(bb_cur)
        bbL_prev = pca.transform(bb_prev)
        vertices_new = np.zeros((vertices.shape[0], 3))
        weights = np.zeros((vertices.shape[0]))
        diff = bbL_cur-bbL_prev
        #blending
        for b in range(bbL_cur.shape[0]):
            temp_weight = 1/np.linalg.norm(vertices-bbL_prev[b,:], axis=1)
            vertices_new[:,0] += diff[b,0]*temp_weight
            vertices_new[:,1] += diff[b,1]*temp_weight
            vertices_new[:,2] += diff[b,2]*temp_weight
            weights += temp_weight
        vertices_new[:,0] /= weights
        vertices_new[:,1] /= weights
        vertices_new[:,2] /= weights
        vertices_new += vertices

        return vertices_new
        #'''
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


    def getOverlapping(self, Parts, PoseBP, bp, RGBD):
        '''
        get the Overlapping region of one body part
        :param Parts: the list of Bodyparts
        :param PoseBP: the list of local to global transform
        :param bp: number of the body part
        :param RGBD: the depth image and intrinsic matrix
        :return: none
        '''
        # get bp vertices
        #initial
        Vtx = Parts[bp].MC.Vertices
        Nmls = Parts[bp].MC.Normales
        Pose = PoseBP[bp]
        size = RGBD.Size
        #transform
        stack_pix = np.ones( (np.size(Vtx[ :,:],0)) , dtype = np.float32)
        stack_pt = np.ones( (np.size(Vtx[ :,:],0)) , dtype = np.float32)
        pix = np.zeros( (np.size(Vtx[ :,:],0),2) , dtype = np.float32)
        pix = np.stack((pix[:,0],pix[:,1],stack_pix),axis = 1)
        pt = np.stack( (Vtx[ :,0],Vtx[ :,1],Vtx[ :,2],stack_pt),axis =1 )
        pt = np.dot(pt,Pose.T)
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ :,:] = np.dot(Nmls[ :,:],Pose[0:3,0:3].T)
        #projection in 2D space
        lpt = np.split(pt,4,axis=1)
        lpt[2] = General.in_mat_zero2one(lpt[2])
        pix[ :,0] = (lpt[0]/lpt[2]).reshape(np.size(Vtx[ :,:],0))
        pix[ :,1] = (lpt[1]/lpt[2]).reshape(np.size(Vtx[ :,:],0))
        pix = np.dot(pix,RGBD.intrinsic.T)
        #get 2d coordinate and index map
        column_index = (np.round(pix[:,0])).astype(int)
        line_index = (np.round(pix[:,1])).astype(int)
        indexmap = np.arange(Vtx.shape[0])
        #create matrix that have 0 when the conditions are not verified and 1 otherwise
        cdt_column = (column_index > -1) * (column_index < size[1])
        cdt_line = (line_index > -1) * (line_index < size[0])
        cdt = cdt_column*cdt_line
        line_index = line_index*cdt
        column_index = column_index*cdt
        #result 
        bp_map = np.zeros((size[0], size[1] ,2), dtype=np.float32)
        bp_map[line_index, column_index,0] = 1*cdt
        bp_map[line_index, column_index,1] = indexmap

        # get the other bp vertices
        bp_n_map = np.zeros((size[0], size[1]), dtype=np.float32)
        for i in range(1,len(PoseBP)):
            if i==bp:
                continue
            #initial
            Vtx = Parts[i].MC.Vertices
            Nmls = Parts[i].MC.Normales
            Pose = PoseBP[i]
            #transform
            stack_pix = np.ones( (np.size(Vtx[ :,:],0)) , dtype = np.float32)
            stack_pt = np.ones( (np.size(Vtx[ :,:],0)) , dtype = np.float32)
            pix = np.zeros( (np.size(Vtx[ :,:],0),2) , dtype = np.float32)
            pix = np.stack((pix[:,0],pix[:,1],stack_pix),axis = 1)
            pt = np.stack( (Vtx[ :,0],Vtx[ :,1],Vtx[ :,2],stack_pt),axis =1 )
            pt = np.dot(pt,Pose.T)
            nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
            nmle[ :,:] = np.dot(Nmls[ :,:],Pose[0:3,0:3].T)
            #projection in 2D space
            lpt = np.split(pt,4,axis=1)
            lpt[2] = General.in_mat_zero2one(lpt[2])
            pix[ :,0] = (lpt[0]/lpt[2]).reshape(np.size(Vtx[ :,:],0))
            pix[ :,1] = (lpt[1]/lpt[2]).reshape(np.size(Vtx[ :,:],0))
            pix = np.dot(pix,RGBD.intrinsic.T)
            #get 2d coordinate and index map
            column_index = (np.round(pix[:,0])).astype(int)
            line_index = (np.round(pix[:,1])).astype(int)
            #create matrix that have 0 when the conditions are not verified and 1 otherwise
            cdt_column = (column_index > -1) * (column_index < size[1])
            cdt_line = (line_index > -1) * (line_index < size[0])
            cdt = cdt_column*cdt_line
            line_index = line_index*cdt
            column_index = column_index*cdt
            #result
            bp_n_map[line_index, column_index] = 1*cdt
        
        index_i, index_y = np.where(1.0*(bp_map[:,:,0]==1)*(bp_n_map==1)==1)
        overlap_index = bp_map[index_i,index_y,1].astype(np.int16)

        #set Vertice and Normals
        if bp==1:
            self.StitchedVertices = self.TransformVtx(Parts[bp].MC.Vertices[overlap_index,:],PoseBP[bp],1)
            self.StitchedNormales = self.TransformNmls(Parts[bp].MC.Normales[overlap_index,:],PoseBP[bp],1)
            
        else:
            PartVertices = self.TransformVtx(Parts[bp].MC.Vertices[overlap_index,:],PoseBP[bp],1)
            PartNormales = self.TransformNmls(Parts[bp].MC.Normales[overlap_index,:],PoseBP[bp],1)
            self.StitchedVertices = np.concatenate((self.StitchedVertices,PartVertices))
            self.StitchedNormales = np.concatenate((self.StitchedNormales,PartNormales))

