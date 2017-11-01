#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 16:47:40 2017

@author: diegothomas, inoeandre
"""

import imp
import numpy as np
from numpy import linalg as LA
from math import sin, cos, acos
import scipy as sp
import pandas 
import warnings

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
                Buffer_B = np.concatenate((Buffer_B, Buffer_B_jun))
                Buffer = np.concatenate((Buffer, Buffer_jun))
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

    def RegisterAllTs(self, NewImage, NewSkeVtx, PreSkeVtx, Vtx_bp):
        '''
        Function that estimate the relative rigid transformations of all bodypart between an input RGB-D images and model(vertex) with pre-frame pose
        :param NewImage: RGBD image
        :param NewSkeVtx: skeleton vertex
        :param PreSkeVtx: skeleton vertex
        :param Vtx_bp: list of vertices list of each body part
        :return: Transform matrixs between Image and the mesh (transform from the pre-frame model to the current frame)
        '''

        # Initialize the Transformation of each body part with the I
        Tr_bp = np.zeros((len(Vtx_bp),4,4), dtype = np.float32)
        Tr_bp[:,0,0] = 1
        Tr_bp[:,1,1] = 1
        Tr_bp[:,2,2] = 1
        Tr_bp[:,3,3] = 1

        #Tr_bp = Tr_bp.reshape((240))
        # sampling
        Vtx_bp_sample = []
        Vtx_bp_sample.append(np.zeros((1,3)))
        for bp in range(1,len(Vtx_bp)):
            sample_index = np.random.randint(Vtx_bp[bp].shape[0], size = int(Vtx_bp[bp].shape[0]/10))
            Vtx_bp_sample.append(Vtx_bp[bp][sample_index,:])

        #print Tr_bp.reshape((len(Vtx_bp),4,4))
        #print (RegisterAllTs_function(Tr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))

        '''
        # all Tr
        #for it in range(self.max_iter[0]):    
        for it in range(1): 
            #res = sp.optimize.least_squares(RegisterAllTs_function, Tr_bp, args=( Vtx[sample_idx,:], Vtx_bp_index[sample_idx],NewImage, NewSkeVtx, PreSkeVtx))
            res = sp.optimize.minimize(RegisterAllTs_function, Tr_bp, args=( Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx), method='Nelder-Mead', options={'maxiter':5000})
            Tr_bp = res.x
        # check
        if res.success:
            print (RegisterAllTs_function(Tr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))
        else:
            print res
            print (RegisterAllTs_function(Tr_bp, Vtx_bp, NewImage, NewSkeVtx, PreSkeVtx))
            print "unsuccessful"
        '''
        bp=1
        RegisterTs_dataterm(Tr_bp[bp,:,:], Vtx_bp[bp], NewImage, NewImage.labels==bp)
        # each Tr
        # data term
        for bp in range(1,len(Vtx_bp)):
            res = sp.optimize.minimize(RegisterTs_dataterm, Tr_bp[bp,:,:], args=( Vtx_bp[bp], NewImage, NewImage.labels==bp), method='Nelder-Mead')
            Tr_bp[bp] = res.x.reshape(4,4)
            if res.success==False:
                print "bp" + str(bp) + " unsuccessful"
        # three term
        for bp in range(1,len(Vtx_bp)):
            res = sp.optimize.minimize(RegisterTs_function, Tr_bp[bp,:,:], args=( Vtx_bp[bp], NewImage, NewSkeVtx, PreSkeVtx, bp, Tr_bp), method='Nelder-Mead')
            Tr_bp[bp] = res.x.reshape(4,4)
            print "bp" + str(bp)
            if res.success==False:
                print "bp" + str(bp) + " unsuccessful"

        return Tr_bp.reshape((len(Vtx_bp),4,4))

import cv2
# energy function
def RegisterTs_dataterm(Tr, MeshVtx, NewRGBD, labels):
    '''
    The data term of ine body part
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
    pt = np.dot(pt, Tr)
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
    #a = labels*1.0
    #a[pix[:,1],pix[:,0]] = 1.0
    #cv2.imshow("",a)
    #cv2.waitKey(1)

    #return term_data
    return sum(term_data)

def RegisterAllTs_function(Tr_bp, MeshVtx_bp, NewRGBD, NewSkeVtx, PreSkeVtx):
    '''
    The energy function with three terms
    :param Tr_bp: the list of Transformation in each body part (type: np array, size: (15, 4, 4))
    :param MeshVtx_bp: the list of the Vtx in body part
    :param NewRGBD: RGBD image
    :retrun: cost list
    '''    
    Tr_bp = Tr_bp.reshape(15,4,4)

    # first term (data term)
    term_data = np.zeros(0)
    for bp in range(1,len(MeshVtx_bp)):
        term_data_bp = RegisterTs_dataterm(Tr_bp[bp], MeshVtx_bp[bp], NewRGBD, NewRGBD.labels==bp)
        term_data = np.concatenate((term_data, term_data_bp))


    # second term(smooth term)
    term_smooth = np.zeros(14)
    Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
    for bp in range(1, 15):
        term_smooth[bp-1] = LA.norm(Tr_bp[bp]-Id4)
    
    # third term(constraint term)
    term_cons = np.zeros(13)
    term_cons[0] = abs(LA.norm(Tr_bp[9,0:3,3]-Tr_bp[10,0:3,3])-LA.norm(NewRGBD.TransfoBB[9][0:3,3]-NewRGBD.TransfoBB[10][0:3,3]))
    term_cons[1] = abs(LA.norm(Tr_bp[2,0:3,3]-Tr_bp[10,0:3,3])-LA.norm(NewRGBD.TransfoBB[2][0:3,3]-NewRGBD.TransfoBB[10][0:3,3]))
    term_cons[2] = abs(LA.norm(Tr_bp[4,0:3,3]-Tr_bp[10,0:3,3])-LA.norm(NewRGBD.TransfoBB[4][0:3,3]-NewRGBD.TransfoBB[10][0:3,3]))
    term_cons[3] = abs(LA.norm(Tr_bp[7,0:3,3]-Tr_bp[10,0:3,3])-LA.norm(NewRGBD.TransfoBB[7][0:3,3]-NewRGBD.TransfoBB[10][0:3,3]))
    term_cons[4] = abs(LA.norm(Tr_bp[5,0:3,3]-Tr_bp[10,0:3,3])-LA.norm(NewRGBD.TransfoBB[5][0:3,3]-NewRGBD.TransfoBB[10][0:3,3]))
    term_cons[5] = abs(LA.norm(Tr_bp[2,0:3,3]-Tr_bp[1,0:3,3])-LA.norm(NewRGBD.TransfoBB[2][0:3,3]-NewRGBD.TransfoBB[1][0:3,3]))
    term_cons[6] = abs(LA.norm(Tr_bp[4,0:3,3]-Tr_bp[3,0:3,3])-LA.norm(NewRGBD.TransfoBB[4][0:3,3]-NewRGBD.TransfoBB[3][0:3,3]))
    term_cons[7] = abs(LA.norm(Tr_bp[1,0:3,3]-Tr_bp[12,0:3,3])-LA.norm(NewRGBD.TransfoBB[1][0:3,3]-NewRGBD.TransfoBB[12][0:3,3]))
    term_cons[8] = abs(LA.norm(Tr_bp[3,0:3,3]-Tr_bp[11,0:3,3])-LA.norm(NewRGBD.TransfoBB[3][0:3,3]-NewRGBD.TransfoBB[11][0:3,3]))
    term_cons[9] = abs(LA.norm(Tr_bp[7,0:3,3]-Tr_bp[8,0:3,3])-LA.norm(NewRGBD.TransfoBB[7][0:3,3]-NewRGBD.TransfoBB[8][0:3,3]))
    term_cons[10] = abs(LA.norm(Tr_bp[5,0:3,3]-Tr_bp[6,0:3,3])-LA.norm(NewRGBD.TransfoBB[5][0:3,3]-NewRGBD.TransfoBB[6][0:3,3]))
    term_cons[11] = abs(LA.norm(Tr_bp[8,0:3,3]-Tr_bp[14,0:3,3])-LA.norm(NewRGBD.TransfoBB[8][0:3,3]-NewRGBD.TransfoBB[14][0:3,3]))
    term_cons[11] = abs(LA.norm(Tr_bp[6,0:3,3]-Tr_bp[13,0:3,3])-LA.norm(NewRGBD.TransfoBB[6][0:3,3]-NewRGBD.TransfoBB[13][0:3,3]))

    # junction term
    term_jun = np.zeros(25)
    stack_jun = np.ones(25, dtype = np.float32) 
    #jun_pre = np.stack((PreSkeVtx[:,0], PreSkeVtx[:,1], PreSkeVtx[:,2], stack_jun), axis=1)
    

    # mix
    term = np.concatenate((term_data*10, term_smooth, term_cons))

    return sum(term_data)


def RegisterTs_function(Tr, MeshVtx, NewRGBD, NewSkeVtx, PreSkeVtx, bp, Tr_bp):
    '''
    The energy function with three terms
    :param Tr: Transformation of one body part 
    :param MeshVtx: the Vtx in body part
    :param NewRGBD: RGBD image
    :param bp: the index of body part
    :param Tr_bp: the list of Transformation in each body part (type: np array, size: (15, 4, 4))
    :retrun: cost list
    '''    
    Tr = Tr.reshape(4,4)

    # first term (data term)
    term_data = RegisterTs_dataterm(Tr, MeshVtx, NewRGBD, NewRGBD.labels==bp)

    # second term(smooth term)
    Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
    term_smooth = LA.norm(Tr-Id4)
    
    # third term(constraint term)
    term_cons = np.zeros(13)
    if bp==1:
        bp_n = [2,12]
    elif bp==2:
        bp_n = [1, 9]
    elif bp==3:
        bp_n = [4, 11]
    elif bp==4:
        bp_n = [3, 9]
    elif bp==5:
        bp_n = [6, 10]
    elif bp==6:
        bp_n = [5, 13]
    elif bp==7:
        bp_n = [8, 10]
    elif bp==8:
        bp_n = [7, 14]
    elif bp==9:
        bp_n = [10]
    elif bp==10:
        bp_n = [2, 4,7,5,9]
    elif bp==11:
        bp_n = [3]
    elif bp==12:
        bp_n = [1]
    elif bp==13:
        bp_n = [6]
    else:
        bp_n = [8]
        
    term_cons = np.zeros(len(bp_n))
    for i in range(len(bp_n)):
        term_cons[i] = abs(LA.norm(Tr[0:3,3]-Tr_bp[bp_n[i],0:3,3])-LA.norm(NewRGBD.TransfoBB[bp][0:3,3]-NewRGBD.TransfoBB[bp_n[i]][0:3,3]))
    
    # junction term
    term_jun = np.zeros(25)
    stack_jun = np.ones(25, dtype = np.float32) 
    #jun_pre = np.stack((PreSkeVtx[:,0], PreSkeVtx[:,1], PreSkeVtx[:,2], stack_jun), axis=1)
    
    # mix
    term = term_data*10+term_smooth+sum(term_cons)

    return term