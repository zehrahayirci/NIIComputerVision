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
from scipy.interpolate import griddata
from scipy.linalg import polar
from pyquaternion import Quaternion
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

    def getJointInfo(self, bp, boneTrans, boneSubTrans):
        """
        get joints' position and transform related to the body part
        :param bp
        :param boneTrans: all bones' transform
        :param boneSubTrans: all parent bone's transform
        :return: bone's DQ, joint's DQ
        """
        #plane3
        sm = 1
        if bp==1:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[0])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[1])
            if sm==1:
                boneSubTrans[0]= np.dot(boneTrans[1], boneSubTrans[0])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[0])
        elif bp==2:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[1])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[14])
            if sm==1:
                boneSubTrans[1]= np.dot(boneTrans[14], boneSubTrans[1])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[1])
        elif bp==3:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[3])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[4])
            if sm==1:
                boneSubTrans[3]= np.dot(boneTrans[4], boneSubTrans[3])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[3])
        elif bp==4:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[4])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[14])
            if sm==1:
                boneSubTrans[4]= np.dot(boneTrans[14], boneSubTrans[4])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[4])
        elif bp==5:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[10])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[15])
            if sm==1:
                boneSubTrans[10]= np.dot(boneTrans[15], boneSubTrans[10])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[10])
        elif bp==6:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[9])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[10])
            if sm==1:
                boneSubTrans[9]= np.dot(boneTrans[10], boneSubTrans[9])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[9])
        elif bp==7:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[7])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[15])
            if sm==1:
                boneSubTrans[7]= np.dot(boneTrans[15], boneSubTrans[7])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[7])
        elif bp==8:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[6])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[7])
            if sm==1:
                boneSubTrans[6]= np.dot(boneTrans[7], boneSubTrans[6])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[6])
        elif bp==9:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[13])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[14])
            if sm==1:
                boneSubTrans[13]= np.dot(boneTrans[14], boneSubTrans[13])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[13])
        elif bp==10:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[15])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[14])
            ####jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[15])
        elif bp==11:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[17])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[3])
            if sm==1:
                boneSubTrans[17]= np.dot(boneTrans[3], boneSubTrans[17])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[17])
        elif bp==12:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[16])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[0])
            if sm==1:
                boneSubTrans[16]= np.dot(boneTrans[0], boneSubTrans[16])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[16])
        elif bp==13:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[18])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[6])
            if sm==1:
                boneSubTrans[18]= np.dot(boneTrans[6], boneSubTrans[18])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[18])
        elif bp==14:
            boneDQ = np.zeros((1,2,4), dtype=np.float32)
            boneDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[19])
            jointDQ = np.zeros((1,2,4), dtype=np.float32)
            jointDQ[0] = General.getDualQuaternionfromMatrix(boneTrans[9])
            if sm==1:
                boneSubTrans[19]= np.dot(boneTrans[9], boneSubTrans[19])
                jointDQ[0] = General.getDualQuaternionfromMatrix(boneSubTrans[19])
        return boneDQ, jointDQ


    def NaiveStitch(self, PartVtx,PartNmls,PartFaces, coordC, coordNew, BBTrNew, boneDQ, jointDQ, planeF, Tg, bp, RGBD=0):
        """
        Add the vertices and faces of each body parts
        together after transforming them in the global coordinates system
        :param PartVtx: List of vertices for a body parts
        :param PartNmls: List of normales for a body parts
        :param PartFaces:  List of faces for a body parts
        :param coordC: the corners of bounding-box in conacial frame
        :param coordNew: the corners of bounding-box in new frame
        :param BBTrNew: the boundong-boxes' transform matrix of new frame
        :param boneTr: the main bone transform
        :param jointTr: the joint transfrom
        :param planeF: the plane function
        :param Tg: the transform matrix from local to global
        :return: none
        """
        #Initialize values from the list of
        ConcatVtx = self.StitchedVertices
        ConcatFaces = self.StitchedFaces
        ConcatNmls = self.StitchedNormales

        # tranform the vertices in the global coordinates system
        PartVertices = self.TransformVtx(PartVtx, coordC, coordNew, BBTrNew,  boneDQ, jointDQ, planeF, Tg, bp, 1, RGBD)
        PartNormales = self.TransformNmls(PartNmls,PartVtx, coordC, coordNew, BBTrNew,  boneDQ, jointDQ, planeF, Tg, bp, 1, RGBD)
        PartFacets = PartFaces + np.max(ConcatFaces)+1

        # concatenation
        self.StitchedVertices = np.concatenate((ConcatVtx,PartVertices))
        self.StitchedNormales = np.concatenate((ConcatNmls,PartNormales))
        self.StitchedFaces = np.concatenate((ConcatFaces,PartFacets))



    def TransformVtx(self, Vtx, coordC, coordNew, BBTrNew, boneDQ, jointDQ, planeF, Tg,bp,  s=1 , RGBD = 0):
        """
        Transform the vertices in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        :param Vtx: List of vertices
        :param coordC: the corners of bounding-box in conacial frame
        :param coordNew: the corners of bounding-box in new frame
        :param BBTrNew: the boundong-boxes' transform matrix of new frame
        :param boneDQ: the main bone DQ
        :param jointDQ: the joint DQ
        :param areaIdx: the index of corner of joint area
        :param Tg: the transform matrix from local to global
        :param s: subsampling factor
        :return: list of transformed vertices
        """
        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        pt = np.dot(pt, Tg.T)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        Vtx = pt[:,0:3]
        newVtx = np.zeros((Vtx.shape[0],3), dtype=np.float32)
        VtxNum = Vtx.shape[0]

        ##joint with plane weight
        if boneDQ.shape[0]==4:
            pt = np.dot(pt, boneDQ.T)
            pt /= pt[:,3].reshape((pt.shape[0], 1))
            Vtx = pt[:,0:3]
        else:
            weights = np.zeros(VtxNum)
            wmap = np.zeros((VtxNum))
            DQnp = np.zeros((VtxNum, 2, 4))
            weightspara = 0.02
            #weight
            weights = np.dot(Vtx,planeF[0:3].T)+planeF[3]
            wmap[:] = weights[:]
            weights = np.exp(-weights*weights/2/weightspara/weightspara)
            weights = weights*(wmap>0)+(wmap<=0)
            #warping
            DQnp += (1-weights).reshape((VtxNum,1,1))*boneDQ[0].reshape((1,2,4))
            DQnp += (weights).reshape((VtxNum,1,1))*jointDQ[0].reshape((1,2,4))
            '''
            #print weights
            Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
            p1 = RGBD.GetProjPts2D_optimize(Vtx, Id4)
            a = np.zeros((RGBD.Size[0], RGBD.Size[1]))
            a[p1[:,1], p1[:,0]] = 1-weights
            cv2.imwrite('C:/Users/CVLab-Yao/Desktop/NIIComputerVision/boundingboxes/'+str(bp)+"_"+str(0)+".png", a*255)
            cv2.imshow("0", a)
            a[p1[:,1], p1[:,0]] = weights
            cv2.imwrite('C:/Users/CVLab-Yao/Desktop/NIIComputerVision/boundingboxes/'+str(bp)+"_"+str(1)+".png", a*255)
            cv2.imshow("1", a)
            cv2.waitKey(0)
            if bp==1:
                self.a = np.zeros((RGBD.Size[0], RGBD.Size[1]))
            self.a[p1[:,1], p1[:,0]] = weights[:]
            '''

            for v in range(VtxNum):
                TrDQ = General.getDualQuaternionNormalize(DQnp[v,:,:])
                Tr = General.getMatrixfromDualQuaternion(TrDQ)
                pt = np.ones(4)
                pt[0:3] = Vtx[v, 0:3]
                pt = np.dot(pt, Tr.T)
                pt /= pt[3]
                Vtx[v,:] = pt[0:3]

        return Vtx

        return newVtx

    def TransformNmls(self, Nmls,  Vtx, coordC, coordNew, BBTrNew, boneDQ, jointDQ, planeF, Tg, bp,  s=1 , RGBD = 0):
        """
        Transform the normales in a system to another system.
        Here it will be mostly used to transform from local system to global coordiantes system
        :param Nmls:  List of normales
        :param Vtx: List of vertices
        :param Tg:  local to global transform
        :param s: subsampling factor
        :return: list of transformed normales
        """
        nmle = np.zeros((Nmls.shape[0], Nmls.shape[1]), dtype = np.float32)
        nmle[ ::s,:] = np.dot(Nmls[ ::s,:],Tg[0:3,0:3].T)


        stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
        pt = np.stack( (Vtx[ ::s,0],Vtx[ ::s,1],Vtx[ ::s,2],stack_pt),axis =1 )
        pt = np.dot(pt, Tg.T)
        pt /= pt[:,3].reshape((pt.shape[0], 1))
        Vtx = pt[:,0:3]
        newVtx = np.zeros((Vtx.shape[0],3), dtype=np.float32)
        VtxNum = Vtx.shape[0]

        ##joint with plane weight
        if boneDQ.shape[0]==4:
            nmle = np.dot(nmle, boneDQ.T[0:3,0:3])
        else:
            weights = np.zeros(VtxNum)
            wmap = np.zeros((VtxNum))
            DQnp = np.zeros((VtxNum, 2, 4))
            weightspara = 0.02
            #weight
            weights = np.dot(Vtx,planeF[0:3].T)+planeF[3]
            wmap[:] = weights[:]
            weights = np.exp(-weights*weights/2/weightspara/weightspara)
            weights = weights*(wmap>0)+(wmap<=0)
            #warping
            DQnp += (1-weights).reshape((VtxNum,1,1))*boneDQ[0].reshape((1,2,4))
            DQnp += (weights).reshape((VtxNum,1,1))*jointDQ[0].reshape((1,2,4))

            for v in range(VtxNum):
                TrDQ = General.getDualQuaternionNormalize(DQnp[v,:,:])
                Tr = General.getMatrixfromDualQuaternion(TrDQ)
                nmle[v] = np.dot(nmle[v], Tr[0:3,0:3].T)

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

    def TransfoBBcorners(self, Pos, BB, BBTrans):
        """
        Transform the corners of bounding-boxes according to the position of the skeleton
        :param Pos: position in 3D of the junctions in Canonical frame
        :param BB: the global bounding-boxes in Canonical frame
        :param BBTrans: the transform matrix of  bounding-boxes in Canonical frame
        :return: the new bounding-boxes in global and the transform matrix of bounding-boxes
        """
        AllrelatedBone = [ [],\
        [[1],[1],[0],[0]], [[1],[14],[14],[1]], \
        [[4],[4],[3],[3]], [[4],[4],[14],[14]], \
        [[15],[15],[10],[10]], [[9],[9],[10],[10]], \
        [[15],[7],[7],[15]], [[6],[6],[7],[7]], \
        [[14],[13],[13],[14]], \
        [[14],[14],[14],[14],[14],[14],[15],[15],[15]], \
        [[3],[3],[17],[17]], [[0],[0],[16],[16]], \
        [[6],[6],[18],[18]], [[9],[9],[19],[19]]
        ]

        newBBs=[]
        newBBs.append(np.array((0,0,0)))
        newBBTrans = []
        newBBTrans.append(np.identity(4))
        for bp in range(1,len(AllrelatedBone)):
            relatedBoneList = AllrelatedBone[bp]
            newBB = np.zeros((len(BB[bp]),3), dtype=np.float32)
            newBBTran = np.zeros((len(BB[bp]),4,4), dtype=np.float32)
            for p in range(len(relatedBoneList)):
                relatedBones = relatedBoneList[p]
                pt = np.array([0.,0.,0.,1.])
                pt[0:3] = BB[bp][p]
                weights = []
                newDQ = np.zeros((2,4), dtype=np.float32)
                scale = 0.0
                for r in range(len(relatedBones)):
                    relatedBone = relatedBones[r]
                    weights.append(1.0)
                    newBBTran[p] += weights[r]*self.boneTrans[relatedBone]
                    tempQR = General.getQuaternionfromMatrix(self.boneTrans[relatedBone])
                    tempT = self.boneTrans[relatedBone][0:3,3]
                    tempDQ = General.getDualQuaternionfromMatrix(self.boneTrans[relatedBone])
                    tempDQ = General.getDualQuaternionNormalize(tempDQ)
                    newDQ = newDQ+tempDQ*weights[r]
                    scale += self.boneTrans[relatedBone][3,3]*weights[r]
                newBBTran[p] /= sum(weights)
                newDQ *= 1/sum(weights)
                scale *= 1/sum(weights)
                newDQ =  General.getDualQuaternionNormalize(newDQ)
                newBBTran[p,:,:] = General.getMatrixfromDualQuaternion(newDQ)
                newBBTran[p,3,3] = scale
                pt = np.dot(newBBTran[p,:,:], pt.T)
                pt /= pt[3]
                newBB[p,:] = pt[0:3]
                newBBTran[p, :, :] = np.dot(newBBTran[p,:,:], BBTrans[bp][p,:,:])
            for p in range(len(relatedBoneList),len(relatedBoneList)*2):
                relatedBones = relatedBoneList[p-len(relatedBoneList)]
                pt = np.array([0.,0.,0.,1.])
                pt[0:3] = BB[bp][p]
                weights = []
                newDQ = np.zeros((2,4), dtype=np.float32)
                scale = 0.0
                for r in range(len(relatedBones)):
                    relatedBone = relatedBones[r]
                    weights.append(1.0)
                    newBBTran[p] += weights[r]*self.boneTrans[relatedBone]
                    tempQR = General.getQuaternionfromMatrix(self.boneTrans[relatedBone])
                    tempT = self.boneTrans[relatedBone][0:3,3]
                    tempDQ = General.getDualQuaternionfromMatrix(self.boneTrans[relatedBone])
                    tempDQ = General.getDualQuaternionNormalize(tempDQ)
                    newDQ = newDQ+tempDQ*weights[r]
                    scale += self.boneTrans[relatedBone][3,3]*weights[r]
                newBBTran[p] /= sum(weights)
                newDQ *= 1/sum(weights)
                scale *= 1/sum(weights)
                newDQ =  General.getDualQuaternionNormalize(newDQ)
                newBBTran[p,:,:] = General.getMatrixfromDualQuaternion(newDQ)
                newBBTran[p,3,3] = scale
                pt = np.dot(newBBTran[p,:,:], pt.T)
                pt /= pt[3]
                newBB[p,:] = pt[0:3]
                newBBTran[p, :, :] = np.dot(newBBTran[p,:,:], BBTrans[bp][p,:,:])
            newBBs.append(newBB)
            newBBTrans.append(newBBTran)
        return newBBs, newBBTrans

    def GetVBonesTrans(self, skeVtx_cur, skeVtx_prev):
        """
        Get transform matrix of bone from previous to current frame
        :param skeVtx_cur: the skeleton Vtx in current frame
        :param skeVtx_prev: the skeleton Vtx in previous frame
        :return: calculated SkeVtx
        """
        bonelist = [[5,6],[4,5],[20,4],[9,10],[8,9],[20,8], \
        [13,14],[12,13],[0,12],[17,18],[16,17],[0,16], \
        [20,2],[2,3],[1,20],[0,1], \
        [6,7], [10,11],[14,15],[18,19]]
        boneorder = [15,14,12,2,5,13,1,0,4,3,8,11,7,10,6,9,16,17,18,19]
        jointParent = [4,20,1,8,20,1,12,0,0,16,0,0,1,20,0,0]
        boneParent = [1,14,14,4,14,14,7,15,8,10,15,11,14,14,15,15,0,3,6,9]
        bonePath = [[15,14,2,1,0], [15,14,2,1], [15,14,2], [15,14,5,4,3], [15,14,5,4], [15,14,5], \
        [8,7,6], [8,7], [8], [11,10, 9], [11,10], [11], \
        [15,14,12], [15,14,12,13], [15,14], [15]]
        bonemodel = np.zeros((20,3))
        for i in range(20):
            if i<=2 or i==8 or i==17:
                bonemodel[i] = np.array([-1.0,0.0,0.0])
            elif i<=5 or i==11 or i==16:
                bonemodel[i] = np.array([1.0,0.0,0.0])
            elif i<13:
                bonemodel[i] = np.array([0.0,1.0,0.0])
            else:
                bonemodel[i] = np.array([0.0,-1.0,0.0])

        # test
        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        tempSkeVtx = copy.deepcopy(skeVtx_prev)

        #initial
        self.boneTrans = np.zeros((20, 4, 4),dtype=np.float32)
        self.boneSubTrans = np.zeros((20, 4, 4),dtype=np.float32)
        self.RTr = np.zeros((20, 4, 4),dtype=np.float32)
        self.STr = np.zeros((20, 4, 4),dtype=np.float32)
        for i in range(20):
            self.boneTrans[i] = np.identity(4,dtype=np.float32)
            self.boneSubTrans[i] = np.identity(4,dtype=np.float32)

        for bIdx in boneorder:
            bone = bonelist[bIdx]
            boneP = bonelist[boneParent[bIdx]]
            v1 = skeVtx_cur[bone[1]]-skeVtx_cur[bone[0]]
            v2 = skeVtx_prev[bone[1]]-skeVtx_prev[bone[0]]
            v3 = skeVtx_prev[boneP[1]] - skeVtx_prev[boneP[0]]
            if bIdx==1:
                v3[0], v3[1] = v3[1], v3[0]
                v3[2] = 0
            elif bIdx==4:
                v3[0], v3[1] = -v3[1], -v3[0]
                v3[2] = 0
            elif bIdx==7 or bIdx==10:
                v3 = skeVtx_prev[boneP[0]]-skeVtx_prev[boneP[1]]
            elif bIdx==14:
                v3 = skeVtx_prev[boneP[1]]-skeVtx_prev[boneP[0]]
            R = General.GetRotatefrom2Vector(v2, v1)
            R_T = np.identity(4)
            R_T[0:3,0:3] = R
            R = General.GetRotatefrom2Vector(v2, v3)
            R1_T = np.identity(4)
            R1_T[0:3,0:3] = R

            S_T = np.identity(4)
            S_T[0:3, 0:3] *= LA.norm(v1)/LA.norm(v2)
            self.boneTrans[bIdx] = R_T
            self.boneSubTrans[bIdx] = R1_T
            T_T = np.identity(4)
            T_T[0:3,3] = skeVtx_prev[bone[0]]
            T1_T = np.identity(4)
            T1_T[0:3,3] = skeVtx_cur[bone[0]]
            self.boneTrans[bIdx] = np.dot(T1_T, np.dot(self.boneTrans[bIdx,:,:],LA.inv(T_T)))
            self.boneSubTrans[bIdx] = np.dot(T_T, np.dot(self.boneSubTrans[bIdx,:,:],LA.inv(T_T)))
            self.RTr[bIdx], self.STr[bIdx] = polar(self.boneTrans[bIdx])

        return self.boneTrans, self.boneSubTrans


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
        ctr = ctr[0:3]/ctr[3]

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
        pt1 = pt1[0:3]/pt1[3]
        pt2 = np.dot(pt2, pose.T)
        pt2 = pt2[0:3]/pt2[3]

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