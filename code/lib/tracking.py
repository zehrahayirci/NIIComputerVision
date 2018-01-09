#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:47:40 2017

@author: diegothomas, inoeandre
"""

import imp
import numpy as np
from numpy import linalg as LA
import math
from math import sin, cos, acos
import scipy as sp
import pandas 
import warnings
import copy

RGBD = imp.load_source('RGBD', './lib/RGBD.py')
General = imp.load_source('General', './lib/General.py')




def Exponential(qsi):
    '''
    This function transform a 6D vector into a 4*4 matrix using Lie's Algebra. It is used to compute the incrementale
    transformation matrix in the tracking process.
    :param qsi: 6D vector
    :return: 4*4 incremental transfo matrix for camera pose estimation
    '''
    theta = LA.norm(qsi[3:6])
    res = np.identity(4)
    
    if (theta != 0.):
        res[0,0] = 1.0 + sin(theta)/theta*0.0 + (1.0 - cos(theta)) / (theta*theta) * (-qsi[5]*qsi[5] - qsi[4]*qsi[4])
        res[1,0] = 0.0 + sin(theta)/theta*qsi[5] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[4])
        res[2,0] = 0.0 - sin(theta)/theta*qsi[4] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[5])
        
        res[0,1] = 0.0 - sin(theta)/theta*qsi[5] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[4])
        res[1,1] = 1.0 + sin(theta) / theta*0.0 + (1.0 - cos(theta))/(theta*theta) * (-qsi[5]*qsi[5] - qsi[3]*qsi[3])
        res[2,1] = 0.0 + sin(theta)/theta*qsi[3] + (1.0 - cos(theta))/(theta*theta) * (qsi[4]*qsi[5])
        
        res[0,2] = 0.0 + sin(theta) / theta*qsi[4] + (1.0 - cos(theta))/(theta*theta) * (qsi[3]*qsi[5])
        res[1,2] = 0.0 - sin(theta)/theta*qsi[3] + (1.0 - cos(theta))/(theta*theta) * (qsi[4]*qsi[5])
        res[2,2] = 1.0 + sin(theta)/theta*0.0 + (1.0 - cos(theta))/(theta*theta) * (-qsi[4]*qsi[4] - qsi[3]*qsi[3])
        
        skew = np.zeros((3,3), np.float64)
        skew[0,1] = -qsi[5]
        skew[0,2] = qsi[4]
        skew[1,0] = qsi[5]
        skew[1,2] = -qsi[3]
        skew[2,0] = -qsi[4]
        skew[2,1] = qsi[3]
        
        V = np.identity(3) + ((1.0-cos(theta))/(theta*theta))*skew + ((theta - sin(theta))/(theta*theta))*np.dot(skew,skew)
        
        res[0,3] = V[0,0]*qsi[0] + V[0,1]*qsi[1] + V[0,2]*qsi[2]
        res[1,3] = V[1,0]*qsi[0] + V[1,1]*qsi[1] + V[1,2]*qsi[2]
        res[2,3] = V[2,0]*qsi[0] + V[2,1]*qsi[1] + V[2,2]*qsi[2]
    else:
        res[0,3] = qsi[0]
        res[1,3] = qsi[1]
        res[2,3] = qsi[2]
        
    return res


def Logarithm(Mat):
    '''
    Inverse of Exponential function. Used to create known transform matrix and test.
    :param Mat: 4*4 transformation matrix
    :return: a 6D vector containing rotation and translation parameters
    '''
    trace = Mat[0,0]+Mat[1,1]+Mat[2,2]
    theta = acos((trace-1.0)/2.0)
    
    qsi = np.array([0.,0.,0.,0.,0.,0.])
    if (theta == 0.):
        qsi[3] = qsi[4] = qsi[5] = 0.0
        qsi[0] = Mat[0,3]
        qsi[1] = Mat[1,3]
        qsi[2] = Mat[2,3]
        return qsi
    
    R = Mat[0:3,0:3]
    lnR = (theta/(2.0*sin(theta))) * (R-np.transpose(R))
    
    qsi[3] = (lnR[2,1] - lnR[1,2])/2.0
    qsi[4] = (lnR[0,2] - lnR[2,0])/2.0
    qsi[5] = (lnR[1,0] - lnR[0,1])/2.0
    
    theta = LA.norm(qsi[3:6])

    skew = np.zeros((3,3), np.float32)
    skew[0,1] = -qsi[5]
    skew[0,2] = qsi[4]
    skew[1,0] = qsi[5]
    skew[1,2] = -qsi[3]
    skew[2,0] = -qsi[4]
    skew[2,1] = qsi[3]
    
    V = np.identity(3) + ((1.0 - cos(theta))/(theta*theta))*skew + ((theta-sin(theta))/(theta*theta))*np.dot(skew,skew)
    V_inv = LA.inv(V)
    
    qsi[0] = V_inv[0,0]*Mat[0,3] + V_inv[0,1]*Mat[1,3] + V_inv[0,2]*Mat[2,3]
    qsi[1] = V_inv[1,0]*Mat[0,3] + V_inv[1,1]*Mat[1,3] + V_inv[1,2]*Mat[2,3]
    qsi[2] = V_inv[2,0]*Mat[0,3] + V_inv[2,1]*Mat[1,3] + V_inv[2,2]*Mat[2,3]
    
    return qsi
    

class Tracker():
    """
    Tracking camera pose class
    """

    def __init__(self, thresh_dist, thresh_norm, lvl, max_iter):
        """
        Constructor
        :param thresh_dist: threshold for distance between vertices
        :param thresh_norm: threshold for distance between normales
        :param lvl:
        :param max_iter: maximum number of iteration
        """
        self.thresh_dist = thresh_dist
        self.thresh_norm = thresh_norm
        self.lvl = lvl
        self.max_iter = max_iter
        


    def RegisterRGBD(self, Image1, Image2):
        '''
        Function that estimate the relative rigid transformation between two input RGB-D images
        :param Image1: First RGBD images
        :param Image2:  Second RGBD images
        :return: Transform matrix between Image1 and Image2
        '''

        # Init
        res = np.identity(4)
        pix = np.array([0., 0., 1.])
        pt = np.array([0., 0., 0., 1.])
        nmle = np.array([0., 0., 0.])


        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                nbMatches = 0
                row = np.array([0.,0.,0.,0.,0.,0.,0.])
                Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                for i in range(Image1.Size[0]/l): # line index (i.e. vertical y axis)
                    for j in range(Image1.Size[1]/l):
                        # Transform current 3D position and normal with current transformation
                        pt[0:3] = Image1.Vtx[i*l,j*l][:]
                        if (LA.norm(pt[0:3]) < 0.1):
                            continue
                        pt = np.dot(res, pt)
                        nmle[0:3] = Image1.Nmls[i*l,j*l][0:3]
                        if (LA.norm(nmle) == 0.):
                            continue
                        nmle = np.dot(res[0:3,0:3], nmle)
                        
                        # Project onto Image2
                        pix[0] = pt[0]/pt[2]
                        pix[1] = pt[1]/pt[2]
                        pix = np.dot(Image2.intrinsic, pix)
                        column_index = int(round(pix[0]))
                        line_index = int(round(pix[1]))
                        
                        if (column_index < 0 or column_index > Image2.Size[1]-1 or line_index < 0 or line_index > Image2.Size[0]-1):
                            continue
                        
                        # Compute distance betwn matches and btwn normals
                        match_vtx = Image2.Vtx[line_index, column_index]
                        distance = LA.norm(pt[0:3] - match_vtx)
                        print "[line,column] : [%d , %d] " %(line_index, column_index)
                        print "match_vtx"
                        print match_vtx
                        print pt[0:3]
                        if (distance > self.thresh_dist):
                            continue
                        
                        match_nmle = Image2.Nmls[line_index, column_index]
                        distance = LA.norm(nmle - match_nmle)
                        print "match_nmle"
                        print match_nmle
                        print nmle                      
                        if (distance > self.thresh_norm):
                            continue
                            
                        w = 1.0
                        # Compute partial derivate for each point
                        row[0] = w*nmle[0]
                        row[1] = w*nmle[1]
                        row[2] = w*nmle[2]
                        row[3] = w*(-match_vtx[2]*nmle[1] + match_vtx[1]*nmle[2])
                        row[4] = w*(match_vtx[2]*nmle[0] - match_vtx[0]*nmle[2])
                        row[5] = w*(-match_vtx[1]*nmle[0] + match_vtx[0]*nmle[1])
                        #current residual
                        row[6] = w*(nmle[0]*(match_vtx[0] - pt[0]) + nmle[1]*(match_vtx[1] - pt[1]) + nmle[2]*(match_vtx[2] - pt[2]))
                                    
                        nbMatches+=1

                        # upper part triangular matrix computation
                        shift = 0
                        for k in range(6):
                            for k2 in range(k,7):
                                Mat[shift] = Mat[shift] + row[k]*row[k2]
                                shift+=1
               
                print ("nbMatches: ", nbMatches)

                # fill up the matrix A.transpose * A and A.transpose * b (A jacobian matrix)
                shift = 0
                for k in range(6):
                    for k2 in range(k,7):
                        val = Mat[shift]
                        shift +=1
                        if (k2 == 6):
                            b[k] = val
                        else:
                            A[k,k2] = A[k2,k] = val
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
        
                #resolve system
                delta_qsi = -LA.tensorsolve(A, b)
                # compute the 4*4 matrix transform
                delta_transfo = LA.inv(Exponential(delta_qsi))

                # update result
                res = np.dot(delta_transfo, res)
                
                print res
        return res
                
    def RegisterRGBD_optimize(self, Image1, Image2):
        '''
        Optimize version of  RegisterRGBD
        :param Image1: First RGBD images
        :param Image2:  Second RGBD images
        :return: Transform matrix between Image1 and Image2
        '''
        res = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
        
        column_index_ref = np.array([np.array(range(Image1.Size[1])) for _ in range(Image1.Size[0])])
        line_index_ref = np.array([x*np.ones(Image1.Size[1], np.int) for x in range(Image1.Size[0])])
        Indexes_ref = column_index_ref + Image1.Size[1]*line_index_ref
        
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                #nbMatches = 0
                #row = np.array([0.,0.,0.,0.,0.,0.,0.])
                #Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((Image1.Size[0]*Image1.Size[1], 6), dtype = np.float32)
                Buffer_B = np.zeros((Image1.Size[0]*Image1.Size[1], 1), dtype = np.float32)
                stack_pix = np.ones((Image1.Size[0], Image1.Size[1]), dtype = np.float32)
                stack_pt = np.ones((np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1)), dtype = np.float32)
                pix = np.zeros((Image1.Size[0], Image1.Size[1],2), dtype = np.float32)
                pix = np.dstack((pix,stack_pix))
                pt = np.dstack((Image1.Vtx[ ::l, ::l, :],stack_pt))
                pt = np.dot(res,pt.transpose(0,2,1)).transpose(1,2,0)
              
                # transform normales
                nmle = np.zeros((Image1.Size[0], Image1.Size[1],Image1.Size[2]), dtype = np.float32)
                nmle[ ::l, ::l,:] = np.dot(res[0:3,0:3],Image1.Nmls[ ::l, ::l,:].transpose(0,2,1)).transpose(1,2,0)

                #Project in 2d space
                #if (pt[2] != 0.0):
                lpt = np.dsplit(pt,4)               
                lpt[2] = General.in_mat_zero2one(lpt[2])
                # if in 1D pix[0] = pt[0]/pt[2]
                pix[ ::l, ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1))
                # if in 1D pix[1] = pt[1]/pt[2]
                pix[ ::l, ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(Image1.Vtx[ ::l, ::l,:],0), np.size(Image1.Vtx[ ::l, ::l,:],1))
                pix = np.dot(Image1.intrinsic,pix[0:Image1.Size[0],0:Image1.Size[1]].transpose(0,2,1)).transpose(1,2,0)

                #checking values are in the frame
                column_index = (np.round(pix[:,:,0])).astype(int)
                line_index = (np.round(pix[:,:,1])).astype(int)                
                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < Image2.Size[1])
                cdt_line = (line_index > -1) * (line_index < Image2.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column

                # Compute distance betwn matches and btwn normals
                diff_Vtx = Image2.Vtx[line_index[:][:], column_index[:][:]] - pt[:,:,0:3]
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=2)
                mask_vtx =  (norm_diff_Vtx < self.thresh_dist)                
                print "mask_vtx"
                print sum(sum(mask_vtx))     
                
                diff_Nmle = Image2.Nmls[line_index[:][:], column_index[:][:]] - nmle        
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=2)
                mask_nmls =  (norm_diff_Nmle < self.thresh_norm)                 
                print "mask_nmls"
                print sum(sum(mask_nmls))   
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=2)
                
                mask_pt =  (pt[:,:,2] > 0.0)

                # Display
                # print "mask_pt"
                # print sum(sum(mask_pt)  )
                #
                # print "cdt_column"
                # print sum(sum( (cdt_column==0))  )
                #
                # print "cdt_line"
                # print sum(sum( (cdt_line==0)) )

                # mask for considering only good value in the linear system
                mask = cdt_line*cdt_column * (pt[:,:,2] > 0.0) * (norm_Norme_Nmle > 0.0) * (norm_diff_Vtx < self.thresh_dist) * (norm_diff_Nmle < self.thresh_norm)
                print "final correspondence"
                print sum(sum(mask))
                

                # partial derivate
                w = 1.0
                Buffer[Indexes_ref[:][:]] = np.dstack((w*mask[:,:]*nmle[ :, :,0], \
                      w*mask[:,:]*nmle[ :, :,1], \
                      w*mask[:,:]*nmle[ :, :,2], \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,1] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,2]), \
                      w*mask[:,:]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2]*nmle[:,:,0] - Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,2]), \
                      w*mask[:,:]*(-Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1]*nmle[:,:,0] + Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0]*nmle[:,:,1]) ))
                #residual
                Buffer_B[Indexes_ref[:][:]] = np.dstack(w*mask[:,:]*(nmle[:,:,0]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,0] - pt[:,:,0])\
                                                                    + nmle[:,:,1]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,1] - pt[:,:,1])\
                                                                    + nmle[:,:,2]*(Image2.Vtx[line_index[:][:], column_index[:][:]][:,:,2] - pt[:,:,2])) ).transpose()

                # Solving sum(A.t * A) = sum(A.t * b) ref newcombe kinect fusion
                # fisrt part of the linear equation
                A = np.dot(Buffer.transpose(), Buffer)
                #second part of the linear equation
                b = np.dot(Buffer.transpose(), Buffer_B).reshape(6)
                
                sign,logdet = LA.slogdet(A)
                det = sign * np.exp(logdet)
                if (det == 0.0):
                    print "determinant null"
                    print det
                    warnings.warn("this is a warning message")
                    break

                # solve equation
                delta_qsi = -LA.tensorsolve(A, b)
                # compute 4*4 matrix
                delta_transfo = Exponential(delta_qsi)
                delta_transfo = General.InvPose(delta_transfo)
                res = np.dot(delta_transfo, res)
                print "delta_transfo"
                print delta_transfo                    
                print "res"
                print res
        return res


            
    def RegisterRGBDMesh(self, NewImage, MeshVtx, MeshNmls,Pose):
        '''
        Function that estimate the relative rigid transformation between an input RGB-D images and a mesh
        :param NewImage: RGBD image
        :param MeshVtx: list of vertices of the mesh
        :param MeshNmls: list of normales of the mesh
        :param Pose:  estimate of the pose of the current image
        :return: Transform matrix between Image1 and the mesh (transform from the first frame to the current frame)
        '''
        res = Pose
        
        line_index = 0
        column_index = 0
        pix = np.array([0., 0., 1.])
        
        pt = np.array([0., 0., 0., 1.])
        nmle = np.array([0., 0., 0.])
        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):
                nbMatches = 0
                row = np.array([0.,0.,0.,0.,0.,0.,0.])
                Mat = np.zeros(27, np.float32)
                b = np.zeros(6, np.float32)
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                for i in range(MeshVtx.shape[0]): # line index (i.e. vertical y axis)
                    # Transform current 3D position and normal with current transformation
                    pt[0:3] = MeshVtx[i][:]
                    if (LA.norm(pt[0:3]) < 0.1):
                        continue
                    pt = np.dot(res, pt)
                    nmle[0:3] = MeshNmls[i][0:3]
                    if (LA.norm(nmle) == 0.):
                        continue
                    nmle = np.dot(res[0:3,0:3], nmle)
                    
                    # Project onto Image2
                    pix[0] = pt[0]/pt[2]
                    pix[1] = pt[1]/pt[2]
                    pix = np.dot(NewImage.intrinsic, pix)
                    column_index = int(round(pix[0]))
                    line_index = int(round(pix[1]))
                    
                    
                    if (column_index < 0 or column_index > NewImage.Size[1]-1 or line_index < 0 or line_index > NewImage.Size[0]-1):
                        continue
                    
                    # Compute distance betwn matches and btwn normals
                    match_vtx = NewImage.Vtx[line_index, column_index]

                    distance = LA.norm(pt[0:3] - match_vtx)
                    if (distance > self.thresh_dist):
                        # print "no Vtx correspondance"
                        # print distance
                        continue
                    match_nmle = NewImage.Nmls[line_index, column_index]

                    distance = LA.norm(nmle - match_nmle)
                    if (distance > self.thresh_norm):

                        # print "no Nmls correspondance"
                        # print distance

                        continue
                        
                    w = 1.0
                    # partial derivate
                    row[0] = w*nmle[0]
                    row[1] = w*nmle[1]
                    row[2] = w*nmle[2]
                    row[3] = w*(-match_vtx[2]*nmle[1] + match_vtx[1]*nmle[2])
                    row[4] = w*(match_vtx[2]*nmle[0] - match_vtx[0]*nmle[2])
                    row[5] = w*(-match_vtx[1]*nmle[0] + match_vtx[0]*nmle[1])
                    # residual
                    row[6] = w*( nmle[0]*(match_vtx[0] - pt[0])\
                               + nmle[1]*(match_vtx[1] - pt[1])\
                               + nmle[2]*(match_vtx[2] - pt[2]))
                                
                    nbMatches+=1
                    # upper part triangular matrix computation
                    shift = 0
                    for k in range(6):
                        for k2 in range(k,7):
                            Mat[shift] = Mat[shift] + row[k]*row[k2]
                            shift+=1
               
                print ("nbMatches: ", nbMatches)

                # fill up the matrix A.transpose * A and A.transpose * b (A jacobian matrix)
                shift = 0
                for k in range(6):
                    for k2 in range(k,7):
                        val = Mat[shift]
                        shift +=1
                        if (k2 == 6):
                            b[k] = val
                        else:
                            A[k,k2] = A[k2,k] = val
                
                det = LA.det(A)
                if (det < 1.0e-10):
                    print "determinant null"
                    break
        
                #solve linear equation
                delta_qsi = -LA.tensorsolve(A, b)
                #compute 4*4 matrix
                delta_transfo = General.InvPose(Exponential(delta_qsi))
                
                res = np.dot(delta_transfo, res)
                
                print res
        return res
    
    
    
    
    def RegisterRGBDMesh_optimize(self, NewImage, NewSkeVtx, PreSkeVtx, MeshVtx, MeshNmls,Pose):
        '''
        Optimize version with CPU  of RegisterRGBDMesh
        :param NewImage: RGBD image
        :param NewSkeVtx: skeleton vertex
        :param PreSkeVtx: skeleton vertex
        :param MeshVtx: list of vertices of the mesh
        :param MeshNmls: list of normales of the mesh
        :return: Transform matrix between Image1 and the mesh (transform from the first frame to the current frame)
        '''
        
        # Initializing the res with the current Pose so that mesh that are in a local coordinates can be
        # transform in the current frame and thus enabling ICP.
        Size = MeshVtx.shape
        res = Pose.copy()
        corres = []        


        for l in range(1,self.lvl+1):
            for it in range(self.max_iter[l-1]):

                # residual matrix
                b = np.zeros(6, np.float32)
                # Jacobian matrix
                A = np.zeros((6,6), np.float32)
                
                # For each pixel find correspondinng point by projection
                Buffer = np.zeros((Size[0], 6), dtype = np.float32)
                Buffer_B = np.zeros((Size[0]), dtype = np.float32)
                stack_pix = np.ones(Size[0], dtype = np.float32) 
                stack_pt = np.ones(np.size(MeshVtx[ ::l,:],0), dtype = np.float32) 
                pix = np.zeros((Size[0], 2), dtype = np.float32)
                pix = np.stack((pix[:,0],pix[:,1],stack_pix), axis = 1)
                pt = np.stack((MeshVtx[ ::l, 0],MeshVtx[ ::l, 1],MeshVtx[ ::l, 2],stack_pt),axis = 1)

                # transform closer vertices to camera pose
                pt = np.dot(res,pt.T).T
                # transform closer normales to camera pose
                nmle = np.zeros((Size[0], Size[1]), dtype = np.float32)
                nmle[ ::l,:] = np.dot(res[0:3,0:3],MeshNmls[ ::l,:].T).T

                # Projection in 2D space
                lpt = np.split(pt,4,axis=1)
                lpt[2] = General.in_mat_zero2one(lpt[2])
                
                # if in 1D pix[0] = pt[0]/pt[2]
                pix[ ::l,0] = (lpt[0]/lpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
                # if in 1D pix[1] = pt[1]/pt[2]
                pix[ ::l,1] = (lpt[1]/lpt[2]).reshape(np.size(MeshVtx[ ::l,:],0))
                pix = np.dot(NewImage.intrinsic,pix[0:Size[0],0:Size[1]].T).T
                column_index = (np.round(pix[:,0])).astype(int)
                line_index = (np.round(pix[:,1])).astype(int)

                # create matrix that have 0 when the conditions are not verified and 1 otherwise
                cdt_column = (column_index > -1) * (column_index < NewImage.Size[1])
                cdt_line = (line_index > -1) * (line_index < NewImage.Size[0])
                line_index = line_index*cdt_line
                column_index = column_index*cdt_column
            
                # compute vtx and nmls differences
                diff_Vtx =  NewImage.Vtx[line_index[:], column_index[:]] - pt[:,0:3] 
                diff_Vtx = diff_Vtx*diff_Vtx
                norm_diff_Vtx = diff_Vtx.sum(axis=1)
                mask_vtx =  (norm_diff_Vtx < self.thresh_dist)
                # print "mask_vtx"
                # print sum(mask_vtx)
                # print "norm_diff_Vtx : max, min , median"
                # print "max : %f; min : %f; median : %f; var :  %f " % (np.max(norm_diff_Vtx),np.min(norm_diff_Vtx) ,np.median(norm_diff_Vtx),np.var(norm_diff_Vtx) )
                
                diff_Nmle = NewImage.Nmls[line_index[:], column_index[:]] - nmle 
                diff_Nmle = diff_Nmle*diff_Nmle
                norm_diff_Nmle = diff_Nmle.sum(axis=1)
                # print "norm_diff_Nmle : max, min , median"
                # print "max : %f; min : %f; median : %f; var :  %f " % (np.max(norm_diff_Nmle),np.min(norm_diff_Nmle) ,np.median(norm_diff_Nmle),np.var(norm_diff_Nmle) )
                
                mask_nmls =  (norm_diff_Nmle < self.thresh_norm)
                # print "mask_nmls"
                # print sum(mask_nmls)
                
                Norme_Nmle = nmle*nmle
                norm_Norme_Nmle = Norme_Nmle.sum(axis=1)

                mask_pt =  (pt[:,2] > 0.0)
                # print "mask_pt"
                # print sum(mask_pt)

                #checking mask
                mask = cdt_line*cdt_column * mask_pt * (norm_Norme_Nmle > 0.0) * mask_vtx * mask_nmls

                # calculate junction cost
                Buffer_jun = np.zeros((25, 6), dtype = np.float32)
                Buffer_B_jun = np.zeros((25), dtype = np.float32)
                for jj in range(1,NewSkeVtx.shape[1]):
                    PVtx = np.ones(4)
                    PVtx[0:3] = PreSkeVtx[0,jj,:]
                    PVtx = np.dot(PVtx,res)
                    w = 100*(NewSkeVtx[0,jj,2]==0)*(PVtx[2]==0)
                    Buffer_jun[jj] = [w,w,w, w*NewSkeVtx[0,jj,1] - w*NewSkeVtx[0,jj,2], w*NewSkeVtx[0,jj,2] - w*NewSkeVtx[0,jj,0], w*NewSkeVtx[0,jj,1] - w*NewSkeVtx[0,jj,0]]
                    Buffer_B[jj] = (NewSkeVtx[0,jj,0]+NewSkeVtx[0,jj,1]+NewSkeVtx[0,jj,2]-PVtx[0]-PVtx[1]-PVtx[2])*w 

                #print "final correspondence"
                #print sum(mask)
                corres.append(sum(mask))

                # partial derivate
                w = 1.0
                Buffer[:] = np.stack((w*mask[:]*nmle[ :,0], \
                      w*mask[:]*nmle[ :, 1], \
                      w*mask[:]*nmle[ :, 2], \
                      w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,1]*nmle[:,2] - NewImage.Vtx[line_index[:], column_index[:]][:,2]*nmle[:,1]), \
                      w*mask[:]*(- NewImage.Vtx[line_index[:], column_index[:]][:,0]*nmle[:,2] + NewImage.Vtx[line_index[:], column_index[:]][:,2]*nmle[:,0] ), \
                      w*mask[:]*(NewImage.Vtx[line_index[:], column_index[:]][:,0]*nmle[:,1] - NewImage.Vtx[line_index[:], column_index[:]][:,1]*nmle[:,0]) ) , axis = 1)
                # residual
                Buffer_B[:] = (w*mask[:]*(nmle[:,0]*(NewImage.Vtx[line_index[:], column_index[:]][:,0] - pt[:,0])\
                                                      + nmle[:,1]*(NewImage.Vtx[line_index[:], column_index[:]][:,1] - pt[:,1])\
                                                      + nmle[:,2]*(NewImage.Vtx[line_index[:], column_index[:]][:,2] - pt[:,2])) )
                # Solving sum(A.t * A) = sum(A.t * b) ref newcombe kinect fusion
                # fisrt part of the linear equation
                #Buffer_B = np.concatenate((Buffer_B, Buffer_B_jun))
                #Buffer = np.concatenate((Buffer, Buffer_jun))
                A = np.dot(Buffer.transpose(), Buffer)
                b = np.dot(Buffer.transpose(), Buffer_B)
                
                sign,logdet = LA.slogdet(A)
                det = sign * np.exp(logdet)
                if (det == 0.0):
                    print "determinant null"
                    print det
                    warnings.warn("this is a warning message")
                    return Pose
                    break

                # solve equation
                delta_qsi = -LA.tensorsolve(A, b)
                # compute 4*4 matrix
                delta_transfo = General.InvPose(Exponential(delta_qsi))

                res = np.dot(delta_transfo, res)
                # print "delta_transfo"
                # print delta_transfo
                # print "res"
                # print res

        if(corres[0]>corres[-1]+corres[0]/20):
            print "correspondence reduce"
            print corres
            return Pose
        
        return res        
    
    def RegisterBBMesh(self, coordsC, coords, VtxList, depthImg, intrinsic, BBTrans_ori, StitchBdy):
        '''
        refine the transform of corner of each body part by registering vertex with depth image
        :param coordsC: the position of corner in first frame
        :param coords: the position of corner in new frame
        :param VtxList: vertices of each body part in global coordinate of first frame
        :param depthImg: depth image
        :param intrinsic: intrinsic matrix
        :param BBTrans_ori: original Bounding-boxes transform matrix
        :StitchBdy: stitchingg object
        :return coords and BBtrans
        '''
        # initial
        BPlist = [12,1,2,11,3,4,13,8,7,14,6,5,9,10] #order
        #BPlist = [10,9,5,6,14,7,8,13,4,3,11,2,1,12] #order
        #BPlist = [1,2,12,3,4,11,5,6,14,7,8,13,10,9] #order
        Id4 = np.identity(4, dtype=np.float32)
        BBTrans = []
        for bp in range(15):
            BBTrans.append(np.zeros((coords[bp].shape[0], 4,4), dtype=np.float32))
            BBTrans[bp][:,0,0] = 1
            BBTrans[bp][:,1,1] = 1
            BBTrans[bp][:,2,2] = 1
            BBTrans[bp][:,3,3] = 1

        # for each body part
        for bp in BPlist:
            print "bp " + str(bp)

            coord = copy.copy(coords[bp])
            for i in range(coords[bp].shape[0]):
                pt = np.array((0.,0.,0.,1.), dtype=np.float32)
                pt[0:3] = coord[i,:]
                coord[i,:] = np.dot(BBTrans[bp][i,:,:], pt.T).T[0:3]
            Vtx = StitchBdy.TransformVtx(VtxList[bp], coordsC[bp], coord, Id4)
            BBTransOpt = Id4
            # energy function
            error1=deformVtx_function(Logarithm(Id4), Vtx, depthImg, intrinsic)
            print error1
            res = sp.optimize.least_squares(deformVtx_function, Logarithm(Id4), max_nfev=6000, args=(Vtx, depthImg, intrinsic))
            #print res
            BBTransOpt = Exponential(res.x)
            error2=deformVtx_function(Logarithm(BBTransOpt), Vtx, depthImg, intrinsic)
            print error2
            if error1<error2:
                BBTransOpt = Id4
            #print BBTransOpt

            for i in range(0,coord.shape[0]):
                BBTrans[bp][i,:,:] = BBTransOpt

            # update same corner point
            if bp==1:
                BBTrans[12][0,:,:] = BBTrans[1][3,:,:]
                BBTrans[12][4,:,:] = BBTrans[1][7,:,:]
                BBTrans[12][1,:,:] = BBTrans[1][2,:,:]
                BBTrans[12][5,:,:] = BBTrans[1][6,:,:]
                if coords[2][0,0]<coords[2][3,0]:
                    BBTrans[2][0,:,:] = BBTrans[1][0,:,:]
                    BBTrans[2][4,:,:] = BBTrans[1][4,:,:]
                    BBTrans[2][3,:,:] = BBTrans[1][1,:,:]
                    BBTrans[2][7,:,:] = BBTrans[1][5,:,:]
                else:
                    BBTrans[2][0,:,:] = BBTrans[1][1,:,:]
                    BBTrans[2][4,:,:] = BBTrans[1][5,:,:]
                    BBTrans[2][3,:,:] = BBTrans[1][0,:,:]
                    BBTrans[2][7,:,:] = BBTrans[1][4,:,:]
            if bp==2:
                if coords[2][0,0]<coords[2][3,0]:
                    BBTrans[1][0,:,:] = BBTrans[2][0,:,:]
                    BBTrans[1][4,:,:] = BBTrans[2][4,:,:]
                    BBTrans[1][1,:,:] = BBTrans[2][3,:,:]
                    BBTrans[1][5,:,:] = BBTrans[2][7,:,:]
                else:
                    BBTrans[1][0,:,:] = BBTrans[2][3,:,:]
                    BBTrans[1][4,:,:] = BBTrans[2][7,:,:]
                    BBTrans[1][1,:,:] = BBTrans[2][0,:,:]
                    BBTrans[1][5,:,:] = BBTrans[2][4,:,:]
                BBTrans[10][0,:,:] = BBTrans[2][2,:,:]
                BBTrans[10][9,:,:] = BBTrans[2][6,:,:]
                BBTrans[10][1,:,:] = BBTrans[2][1,:,:]
                BBTrans[10][10,:,:] = BBTrans[2][5,:,:]
            if bp==3:
                BBTrans[11][0,:,:] = BBTrans[3][3,:,:]
                BBTrans[11][4,:,:] = BBTrans[3][7,:,:]
                BBTrans[11][1,:,:] = BBTrans[3][2,:,:]
                BBTrans[11][5,:,:] = BBTrans[3][6,:,:]
                if coords[4][1,0]<coords[4][0,0]:
                    BBTrans[4][1,:,:] = BBTrans[3][0,:,:]
                    BBTrans[4][5,:,:] = BBTrans[3][4,:,:]
                    BBTrans[4][0,:,:] = BBTrans[3][1,:,:]
                    BBTrans[4][4,:,:] = BBTrans[3][5,:,:]
                else:
                    BBTrans[4][1,:,:] = BBTrans[3][1,:,:]
                    BBTrans[4][5,:,:] = BBTrans[3][5,:,:]
                    BBTrans[4][0,:,:] = BBTrans[3][0,:,:]
                    BBTrans[4][4,:,:] = BBTrans[3][4,:,:]
            if bp==4:
                if coords[4][1,0]<coords[4][0,0]:
                    BBTrans[3][0,:,:] = BBTrans[4][1,:,:]
                    BBTrans[3][4,:,:] = BBTrans[4][5,:,:]
                    BBTrans[3][1,:,:] = BBTrans[4][0,:,:]
                    BBTrans[3][5,:,:] = BBTrans[4][4,:,:]
                else:
                    BBTrans[3][0,:,:] = BBTrans[4][0,:,:]
                    BBTrans[3][4,:,:] = BBTrans[4][4,:,:]
                    BBTrans[3][1,:,:] = BBTrans[4][1,:,:]
                    BBTrans[3][5,:,:] = BBTrans[4][5,:,:]
                BBTrans[10][4,:,:] = BBTrans[4][3,:,:] 
                BBTrans[10][13,:,:] = BBTrans[4][7,:,:] 
                BBTrans[10][5,:,:] = BBTrans[4][2,:,:] 
                BBTrans[10][14,:,:] = BBTrans[4][6,:,:] 
            if bp==5:
                BBTrans[6][2,:,:] = BBTrans[5][3,:,:]
                BBTrans[6][6,:,:] = BBTrans[5][7,:,:]
                BBTrans[6][3,:,:] = BBTrans[5][2,:,:]
                BBTrans[6][7,:,:] = BBTrans[5][6,:,:]
                BBTrans[10][7,:,:] = BBTrans[5][0,:,:]
                BBTrans[10][16,:,:] = BBTrans[5][4,:,:]
                BBTrans[10][6,:,:] = BBTrans[5][1,:,:]
                BBTrans[10][15,:,:] = BBTrans[5][5,:,:]
                BBTrans[7][0,:,:] = BBTrans[5][0,:,:]
                BBTrans[7][4,:,:] = BBTrans[5][4,:,:]
            if bp==6:
                BBTrans[5][3,:,:] = BBTrans[6][2,:,:]
                BBTrans[5][7,:,:] = BBTrans[6][5,:,:]
                BBTrans[5][2,:,:] = BBTrans[6][3,:,:]
                BBTrans[5][6,:,:] = BBTrans[6][7,:,:]
                BBTrans[14][0,:,:] = BBTrans[6][1,:,:]
                BBTrans[14][4,:,:] = BBTrans[6][5,:,:]
                BBTrans[14][1,:,:] = BBTrans[6][0,:,:]
                BBTrans[14][5,:,:] = BBTrans[6][4,:,:]
            if bp==7:
                BBTrans[8][2,:,:] = BBTrans[7][2,:,:]
                BBTrans[8][6,:,:] = BBTrans[7][6,:,:]
                BBTrans[8][3,:,:] = BBTrans[7][1,:,:]
                BBTrans[8][7,:,:] = BBTrans[7][5,:,:]
                BBTrans[10][7,:,:] = BBTrans[7][0,:,:]
                BBTrans[10][16,:,:] = BBTrans[7][4,:,:]
                BBTrans[10][8,:,:] = BBTrans[7][3,:,:]
                BBTrans[10][17,:,:] = BBTrans[7][7,:,:]
                BBTrans[5][0,:,:] = BBTrans[7][0,:,:]
                BBTrans[5][4,:,:] = BBTrans[7][4,:,:]
            if bp==8:
                BBTrans[7][2,:,:] = BBTrans[8][2,:,:]
                BBTrans[7][6,:,:] = BBTrans[8][6,:,:]
                BBTrans[7][1,:,:] = BBTrans[8][3,:,:]
                BBTrans[7][5,:,:] = BBTrans[8][7,:,:]
                BBTrans[13][0,:,:] = BBTrans[8][1,:,:]
                BBTrans[13][4,:,:] = BBTrans[8][5,:,:]
                BBTrans[13][1,:,:] = BBTrans[8][0,:,:]
                BBTrans[13][5,:,:] = BBTrans[8][4,:,:]
            if bp==9:
                BBTrans[10][2,:,:] = BBTrans[9][0,:,:]
                BBTrans[10][11,:,:] = BBTrans[9][4,:,:]
                BBTrans[10][3,:,:] = BBTrans[9][3,:,:]
                BBTrans[10][12,:,:] = BBTrans[9][7,:,:]
            if bp==10:
                BBTrans[2][2,:,:] = BBTrans[10][0,:,:]
                BBTrans[2][6,:,:] = BBTrans[10][9,:,:]
                BBTrans[2][1,:,:] = BBTrans[10][1,:,:]
                BBTrans[2][5,:,:] = BBTrans[10][10,:,:]
                BBTrans[9][0,:,:] = BBTrans[10][2,:,:]
                BBTrans[9][4,:,:] = BBTrans[10][11,:,:]
                BBTrans[9][3,:,:] = BBTrans[10][3,:,:]
                BBTrans[9][7,:,:] = BBTrans[10][12,:,:]
                BBTrans[4][3,:,:] = BBTrans[10][4,:,:]
                BBTrans[4][7,:,:] = BBTrans[10][13,:,:]
                BBTrans[4][2,:,:] = BBTrans[10][5,:,:]
                BBTrans[4][6,:,:] = BBTrans[10][14,:,:]
                BBTrans[5][1,:,:] = BBTrans[10][6,:,:]
                BBTrans[5][5,:,:] = BBTrans[10][15,:,:]
                BBTrans[5][0,:,:] = BBTrans[10][7,:,:]
                BBTrans[5][4,:,:] = BBTrans[10][16,:,:]
                BBTrans[7][0,:,:] = BBTrans[10][7,:,:]
                BBTrans[7][4,:,:] = BBTrans[10][16,:,:]
                BBTrans[7][3,:,:] = BBTrans[10][8,:,:]
                BBTrans[7][7,:,:] = BBTrans[10][17,:,:]
            if bp==11:
                BBTrans[3][3,:,:] = BBTrans[11][0,:,:]
                BBTrans[3][7,:,:] = BBTrans[11][4,:,:]
                BBTrans[3][2,:,:] = BBTrans[11][1,:,:]
                BBTrans[3][6,:,:] = BBTrans[11][5,:,:]
            if bp==12:
                BBTrans[1][3,:,:] = BBTrans[12][0,:,:]
                BBTrans[1][7,:,:] = BBTrans[12][4,:,:]
                BBTrans[1][2,:,:] = BBTrans[12][1,:,:]
                BBTrans[1][6,:,:] = BBTrans[12][5,:,:]
            if bp==13:
                BBTrans[8][1,:,:] = BBTrans[13][0,:,:]
                BBTrans[8][5,:,:] = BBTrans[13][4,:,:]
                BBTrans[8][0,:,:] = BBTrans[13][1,:,:]
                BBTrans[8][4,:,:] = BBTrans[13][5,:,:]
            if bp==14:
                BBTrans[6][1,:,:] = BBTrans[14][0,:,:]
                BBTrans[6][5,:,:] = BBTrans[14][4,:,:]
                BBTrans[6][0,:,:] = BBTrans[14][1,:,:]
                BBTrans[6][4,:,:] = BBTrans[14][5,:,:]
        # merge to BBTrans_ori and modified coords
        for bp in BPlist:
            for i in range(coords[bp].shape[0]):
                BBTrans_ori[bp][i,:,:] = np.dot(BBTrans[bp][i,:,:], BBTrans_ori[bp][i,:,:])
                pt = np.array((0.,0.,0.,1.), dtype=np.float32)
                pt[0:3] = coords[bp][i,:]
                coords[bp][i,:] = np.dot(BBTrans[bp][i,:,:], pt.T).T[0:3]

        return coords, BBTrans_ori

    def RegisterBB(self, bb_cur, bb_prev):
        '''
        Function that estimate the relative rigid transformation between corners of two boundingboxes of the same body part
        :param bb_cur: the bounding-boxes of current frame in one body part
        :param bb_prev: the bounding-boxes of previous frame in one body part
        :return: Transform matrix between two BB
        '''
        # initial
        transfo = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        A = np.zeros((bb_cur.shape[0]*3,12))
        b = np.zeros((bb_cur.shape[0]*3,1))
        for i in range(bb_cur.shape[0]):
            A[i*3,0:3] = bb_prev[i,:]
            A[i*3, 9] = 1
            A[i*3+1, 3:6] = bb_prev[i,:]
            A[i*3+1, 10] = 1
            A[i*3+2, 6:9] = bb_prev[i,:]
            A[i*3+2, 11] = 1
            b[i*3] = bb_cur[i,0]
            b[i*3+1] = bb_cur[i,1]
            b[i*3+2] = bb_cur[i,2]
        # Ax=b
        res = np.linalg.lstsq(A, b)
        print "residual c="
        print res[1]
        # reshape
        x = res[0]
        transfo[0,0] = x[0]
        transfo[0,1] = x[1]
        transfo[0,2] = x[2]
        transfo[1,0] = x[3]
        transfo[1,1] = x[4]
        transfo[1,2] = x[5]
        transfo[2,0] = x[6]
        transfo[2,1] = x[7]
        transfo[2,2] = x[8]
        transfo[0,3] = x[9]
        transfo[1,3] = x[10]
        transfo[2,3] = x[11]

        return transfo
        

    def RegisterAllTs(self, NewImage, NewSkeVtx, PreSkeVtx, Vtx_bp):
        '''
        Function that estimate the relative rigid transformations of all bodypart between an input RGB-D images and model(vertex) with pre-frame pose
        :param NewImage: RGBD image
        :param NewSkeVtx: skeleton vertex
        :param PreSkeVtx: skeleton vertex
        :param Vtx_bp: list of vertices list of each body part
        :return: Transform matrixs between Image and the mesh (transform from the pre-frame model to the current frame)
        '''

        # Initialize the Transformation of each body part with the skeleton
        Tr_bp = np.zeros((len(Vtx_bp),4,4), dtype = np.float32)
        Tr_bp[:,0,0] = 1
        Tr_bp[:,1,1] = 1
        Tr_bp[:,2,2] = 1
        Tr_bp[:,3,3] = 1
        for bp in range(1,len(Vtx_bp)):
            p_n = General.getBodypartPoseIndex(bp)
            meanSkeTran = np.mean(NewSkeVtx[0,p_n,:] - PreSkeVtx[0,p_n,:], axis=0)
            Tr_bp[bp][0:3,3] = meanSkeTran
        initTr_bp = Tr_bp 
        
        #Tr_bp = Tr_bp.reshape((240))
        # sampling
        Vtx_bp_sample = []
        Vtx_bp_sample.append(np.zeros((1,3)))
        for bp in range(1,len(Vtx_bp)):
            sample_index = np.random.randint(Vtx_bp[bp].shape[0], size = int(Vtx_bp[bp].shape[0]/10))
            Vtx_bp_sample.append(Vtx_bp[bp][sample_index,:])

        #print Tr_bp.reshape((len(Vtx_bp),4,4))
        print (RegisterAllTs_function(Tr_bp, initTr_bp,Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))

        '''
        # all Tr
        #for it in range(self.max_iter[0]):    
        for it in range(1): 
            #res = sp.optimize.least_squares(RegisterAllTs_function, Tr_bp, args=( initTr_bp, Vtx[sample_idx,:], Vtx_bp_index[sample_idx],NewImage, NewSkeVtx, PreSkeVtx))
            res = sp.optimize.minimize(RegisterAllTs_function, Tr_bp, args=( initTr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx), method='Nelder-Mead')
            Tr_bp = res.x
        # check
        if res.success:
            print (RegisterAllTs_function(Tr_bp, initTr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))
        else:
            print res
            print (RegisterAllTs_function(Tr_bp, initTr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))
            print "unsuccessful"
        '''
        
        # each Tr
        # data term
        for bp in range(1,len(Vtx_bp)):
            res = sp.optimize.minimize(RegisterTs_dataterm, Tr_bp[bp,:,:], args=( Vtx_bp[bp], NewImage, NewImage.labels>0), method='Nelder-Mead')
            Tr_bp[bp] = res.x.reshape(4,4)
            if res.success==False:
                print "bp" + str(bp) + " unsuccessful"    
        
        # three term
        for t in range(1):
            for bp in range(1,len(Vtx_bp)):
                res = sp.optimize.minimize(RegisterTs_function, Tr_bp[bp,:,:], args=( initTr_bp[bp], Vtx_bp[bp], NewImage, NewSkeVtx, PreSkeVtx, bp, Tr_bp), method='Nelder-Mead')
                Tr_bp[bp] = res.x.reshape(4,4)
                if res.success==False:
                    print "bp" + str(bp) + " unsuccessful"
        #'''

        print (RegisterAllTs_function(Tr_bp, initTr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))

        return Tr_bp.reshape((len(Vtx_bp),4,4))

import cv2
# energy function

def RegisterTs_dataterm(Tr, MeshVtx, NewRGBD, labels):
    '''
    The data term of one body part
    :param Tr: Transformation of one body part 
    :param MeshVtx: the Vtx in body part
    :param NewRGBD: RGBD image
    :param labels: the label of body part 
    :retrun: cost list
    '''  
    Tr = Tr.reshape(4,4)

    # get the 2Dmap of 3D point of new depth image
    ImgVtx = NewRGBD.Vtx

    # sample
    sample_idx = np.random.randint(MeshVtx.shape[0], size = int(MeshVtx.shape[0]/10))
    sample_idx = np.arange(MeshVtx.shape[0])
    #MeshVtx = MeshVtx[sample_idx,:]

    size = MeshVtx.shape
    # find the correspondence 2D point by projection 
    stack_pix = np.ones(size[0], dtype = np.float32) 
    stack_pt = np.ones(size[0], dtype = np.float32) 
    pix = np.zeros((size[0], 2), dtype = np.float32) 
    pix = np.stack((pix[:,0],pix[:,1],stack_pix), axis = 1)
    pt = np.stack((MeshVtx[:,0],MeshVtx[:,1],MeshVtx[:,2],stack_pt),axis = 1)
    # transform vertices to camera pose
    pt = np.dot(Tr, pt.T).T
    # project to 2D coordinate
    lpt = np.split(pt, 4, axis=1)
    lpt[2] = General.in_mat_zero2one(lpt[2])
    # pix[0] = pt[0]/pt[2]
    pix[:,0] = (lpt[0]/lpt[2]).reshape(size[0])
    pix[:,1] = (lpt[1]/lpt[2]).reshape(size[0])
    pix = np.dot(NewRGBD.intrinsic,pix.T).T.astype(np.int16)
    mask = (pix[:,0]>=0) * (pix[:,0]<NewRGBD.Size[1]) * (pix[:,1]>=0) * (pix[:,1]<NewRGBD.Size[0])
    mask = mask * (ImgVtx[pix[:,1]*mask,pix[:,0]*mask, 2]>0)
    # get data term
    term_data = (pt[:,0:3]-ImgVtx[pix[:,1]*mask, pix[:,0]*mask,:])
    term_data = LA.norm(term_data, axis=1)*mask
    term_data[labels[pix[:,1]*mask,pix[:,0]*mask]==0] = 0.5

    # testing
    #a = labels*0.3
    #a[pix[:,1],pix[:,0]] += 0.7
    #cv2.imshow("",a)
    #cv2.waitKey(1)

    #return term_data
    return sum(term_data)

def RegisterTs_constraintterm(Tr, NewRGBD, bp, Tr_bp):
    '''
    The constraint term of one body part
    :param Tr: Transformation of one body part 
    :param NewRGBD: RGBD image
    :param bp: the index of body part
    :param Tr_bp: the list of Transformation in each body part (type: np array, size: (15, 4, 4))
    :retrun: cost list
    '''  

    bp_n = General.getConnectBP(bp)
    term_cons = np.zeros(len(bp_n))
    for i in range(len(bp_n)):
        term_cons[i] = abs(LA.norm(Tr[0:3,3]-Tr_bp[bp_n[i],0:3,3])-LA.norm(NewRGBD.TransfoBB[bp][0:3,3]-NewRGBD.TransfoBB[bp_n[i]][0:3,3]))
        #term_cons[i] = abs(LA.norm(Tr[0:3,3]-Tr_bp[bp_n[i],0:3,3]+NewRGBD.TransfoBB[bp][0:3,3]-NewRGBD.TransfoBB[bp_n[i]][0:3,3]))
    
    #return term_cons
    return sum(term_cons)


def RegisterAllTs_function(Tr_bp, initTr_bp, MeshVtx_bp, NewRGBD, NewSkeVtx, PreSkeVtx):
    '''
    The energy function with three terms
    :param Tr_bp: the list of Transformation in each body part (type: np array, size: (15, 4, 4))
    :param MeshVtx_bp: the list of the Vtx in body part
    :param NewRGBD: RGBD image
    :retrun: cost list
    '''    
    Tr_bp = Tr_bp.reshape(15,4,4)

    # first term (data term)
    #term_data = np.zeros(0)
    term_data = 0
    for bp in range(1,len(MeshVtx_bp)):
        term_data_bp = RegisterTs_dataterm(Tr_bp[bp], MeshVtx_bp[bp], NewRGBD, NewRGBD.labels>0)
        #term_data = np.concatenate((term_data, term_data_bp))
        term_data = term_data+term_data_bp


    # second term(smooth term)
    term_smooth = np.zeros(14) 
    for bp in range(1, 15):
        term_smooth[bp-1] = LA.norm(Tr_bp[bp]-initTr_bp[bp])
    
    # third term(constraint term)
    term_cons = np.zeros(26)
    t = 0
    for bp in range(1, len(MeshVtx_bp)):
        term_cons_bp = RegisterTs_constraintterm(Tr_bp[bp], NewRGBD, bp, Tr_bp)
        #term_cons[t:t+term_cons_bp.shape[0]] = term_cons_bp
        term_cons[bp] = term_cons_bp
        #t += term_cons_bp.shape[0]

    # junction term
    term_jun = np.zeros(25)
    stack_jun = np.ones(25, dtype = np.float32) 
    #jun_pre = np.stack((PreSkeVtx[:,0], PreSkeVtx[:,1], PreSkeVtx[:,2], stack_jun), axis=1)
    

    # mix
    #term = np.concatenate((term_data*0.001, term_smooth, term_cons))
    term = (term_data*0.001)+sum(term_smooth)+sum(term_cons)

    return term


def RegisterTs_function(Tr, initTr, MeshVtx, NewRGBD, NewSkeVtx, PreSkeVtx, bp, Tr_bp):
    '''
    The energy function with three terms
    :param Tr: Transformation of one body part 
    :param initTr: the initialized Transformation of one body part 
    :param MeshVtx: the Vtx in body part
    :param NewRGBD: RGBD image
    :param bp: the index of body part
    :param Tr_bp: the list of Transformation in each body part (type: np array, size: (15, 4, 4))
    :retrun: cost list
    '''    
    Tr = Tr.reshape(4,4)

    # first term (data term)
    term_data = RegisterTs_dataterm(Tr, MeshVtx, NewRGBD, NewRGBD.labels>0)

    # second term(smooth term)
    #Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
    term_smooth = LA.norm(Tr-initTr)
    
    # third term(constraint term)
    term_cons = RegisterTs_constraintterm(Tr, NewRGBD, bp, Tr_bp)
    
    # junction term
    term_jun = np.zeros(25)
    stack_jun = np.ones(25, dtype = np.float32) 
    #jun_pre = np.stack((PreSkeVtx[:,0], PreSkeVtx[:,1], PreSkeVtx[:,2], stack_jun), axis=1)
    
    # mix
    term = term_data*0.001+term_smooth+term_cons

    return term

def deformVtx_function(Qsi, Vtx, depthImg, intrinsic):
    '''
    :param Qsi: Logarithm(transform matrix of one bounding-box)
    :param Vtx: the vertice in global at first frame
    :param depthImg: the depth Image
    :param intrinsic: intrinsic matrix
    '''
    # 1D->3D
    BBTrans = Exponential(Qsi)
    # normalize
    #BBTrans[3,0:3] = 0
    #BBTrans[3,3] = 1

    # transform
    stack_pt = np.ones(np.size(Vtx,0), dtype = np.float32)
    pt = np.stack( (Vtx[:,0],Vtx[:,1],Vtx[:,2],stack_pt),axis =1 )
    pt = np.dot(pt, BBTrans.T)
    Vtx = pt[:,0:3]
    Vtxold = Vtx
    
    # project to 2D
    pix = np.array([0., 0., 1.])
    pix = np.stack((pix for i in range(len(Vtx)) ))
    Vtx[:,2] = General.in_mat_zero2one(Vtx[:,2])
    pix[:,0] = Vtx[:,0]/Vtx[:,2]
    pix[:,1] = Vtx[:,1]/Vtx[:,2]
    pix = np.dot( pix, intrinsic.T)
    pix = pix.astype(int)

    # get error
    bmap = (pix[:,0]>=0)*(pix[:,0]<depthImg.shape[1])*(pix[:,1]>=0)*(pix[:,1]<depthImg.shape[0])
    error = Vtxold[:,2]-depthImg[pix[:,1]*bmap, pix[:,0]*bmap]
    error = np.abs(error*error)
    error = sum(error*bmap+sum(bmap==0)*4)
    
    a = depthImg>0
    a = a*0.5
    a[pix[:,1].astype(int)*bmap, pix[:,0].astype(int)*bmap] +=0.5
    cv2.imshow("", a)
    cv2.waitKey(1)
    
    return error