"""
File to handle program main loop
"""

# File created by Diego Thomas the 21-11-2016
# Second Author Inoe ANDRE

#
import cv2
from math import cos,sin
import numpy as np
import Tkinter as tk
from PIL import Image, ImageTk
import imp
import scipy.io
import time
from skimage.draw import line_aa
from plyfile import PlyData, PlyElement

RGBD = imp.load_source('RGBD', './lib/RGBD.py')
TrackManager = imp.load_source('TrackManager', './lib/tracking.py')
TSDFtk = imp.load_source('TSDFtk', './lib/TSDF.py')
GPU = imp.load_source('GPUManager', './lib/GPUManager.py')
My_MC = imp.load_source('My_MarchingCube', './lib/My_MarchingCube.py')
Stitcher = imp.load_source('Stitcher', './lib/Stitching.py')
BdyPrt = imp.load_source('BodyParts', './lib/BodyParts.py')
General = imp.load_source('General', './lib/General.py')


class Application(tk.Frame):
    """
    Class to apply the segmented fusion
    It also contains function to handle keyboard and mouse inputs
    """


    def key(self, event):
        """
        Function to handle keyboard inputs
        :param event: press a button on the keyboard
        :return: none
        """
        Transfo = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])

        if (event.keysym == 'Escape'):
            self.root.destroy()
        if (event.keysym == 'd'):
            Transfo[0,3] = -0.1
        if (event.keysym == 'a'):
            Transfo[0,3] = 0.1
        if (event.keysym == 'w'):
            Transfo[1,3] = 0.1
        if (event.keysym == 's'):
            Transfo[1,3] = -0.1
        if (event.keysym == 'e'):
            Transfo[2,3] = 0.1
        if (event.keysym == 'q'):
            Transfo[2,3] = -0.1
        if (event.keysym == 'c'):
            self.color_tag = (self.color_tag+1) %2
        if(event.keysym == 'u'):
            self.skeleton_tag = ~self.skeleton_tag
        if(event.keysym == 'i'):
            self.center_tag = ~self.center_tag
        if(event.keysym == 'o'):
            self.Sys_tag = ~self.Sys_tag
        if(event.keysym == 'p'):
            self.OBBox_tag = ~self.OBBox_tag
        if(event.keysym == 'l'):
            self.first_tag = ~self.first_tag

        if (event.keysym != 'Escape'):
            self.Pose = np.dot(Transfo, self.Pose)
            rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            if(self.first_tag):
                rendering = self.RGBD[0].Draw_optimize(rendering,self.Pose, self.w.get(), self.color_tag)
            else:
                # Projection for each body parts done separately
                PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
                for bp in range(1,len(self.Parts)):
                    bou = bp
                    for i in range(4):
                        for j in range(4):
                            PoseBP[i][j] = self.Parts[bou].Tlg[i][j]
                    PoseBP = np.dot(self.Pose, PoseBP)
                    rendering = self.RGBD[0].DrawMesh(rendering,self.Parts[bou].MC.Vertices,self.Parts[bou].MC.Normales,PoseBP, self.w.get(), self.color_tag)

            img = Image.fromarray(rendering, 'RGB')
            self.imgTk=ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
            if(self.skeleton_tag):
                self.DrawSkeleton2D(self.Pose)
            if(self.center_tag):
                self.DrawCenters2D(self.Pose)
            if(self.Sys_tag):
                self.DrawCenters2D(self.Pose)
                self.DrawSys2D(self.Pose)
            if(self.OBBox_tag):
                self.DrawOBBox2D(self.Pose)



    def mouse_press(self, event):
        """
        Function to handle mouse press event, displacement related to the event
        :param event: a click with the mouse
        :return: none

        """
        self.x_init = event.x
        self.y_init = event.y

    def mouse_release(self, event):
        """
        Function to handle mouse release events, displacement related to the event
        :param event: a click with the mouse
        :return: none
        """
        x = event.x
        y = event.y



    def mouse_motion(self, event):
        """
        Function to handle mouse motion events. displacement related to the event
        :param event: moving mouse when a button is pressed
        :return: none
        """
        if (event.y < 426):
            delta_x = event.x - self.x_init
            delta_y = event.y - self.y_init

            angley = 0.
            if (delta_x > 0.):
                angley = -0.02
            elif (delta_x < 0.):
                angley = 0.02 #pi * 2. * delta_x / float(self.Size[0])
            RotY = np.array([[cos(angley), 0., sin(angley), 0.], \
                             [0., 1., 0., 0.], \
                             [-sin(angley), 0., cos(angley), 0.], \
                             [0., 0., 0., 1.]])
            #self.Pose = np.dot(self.Pose, RotY)

            anglex = 0.
            if (delta_y > 0.):
                anglex = 0.02
            elif (delta_y < 0.):
                anglex = -0.02 # pi * 2. * delta_y / float(self.Size[0])
            RotX = np.array([[1., 0., 0., 0.], \
                            [0., cos(anglex), -sin(anglex), 0.], \
                            [0., sin(anglex), cos(anglex), 0.], \
                            [0., 0., 0., 1.]])
            Transla_in = np.array([[1., 0., 0., -self.Pose[0][3]-self.RGBD[0].ctr3D[11][0]], [0., 1., 0., -self.Pose[1][3]-self.RGBD[0].ctr3D[11][1]], [0., 0., 1., -self.Pose[2][3]-self.RGBD[0].ctr3D[11][2]], [0., 0., 0., 1.]])
            Transla = np.array([[1., 0., 0., self.Pose[0][3]+self.RGBD[0].ctr3D[11][0]], [0., 1., 0., +self.Pose[1][3]+self.RGBD[0].ctr3D[11][1]], [0., 0., 1., +self.Pose[2][3]+self.RGBD[0].ctr3D[11][2]], [0., 0., 0., 1.]])
            #self.Pose = np.dot(self.Pose, RotX)
            self.Pose = np.dot(Transla, np.dot(RotX, np.dot(RotY, np.dot(Transla_in,self.Pose))))

        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        if(self.first_tag):
            rendering = self.RGBD[0].Draw_optimize(rendering,self.Pose, self.w.get(), self.color_tag)
        else:
            # Projection for each body parts done separately
            PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
            for bp in range(1,len(self.Parts)):
                bou = bp
                for i in range(4):
                    for j in range(4):
                        PoseBP[i][j] = self.Parts[bou].Tlg[i][j]
                PoseBP = np.dot(self.Pose, PoseBP)
                rendering = self.RGBD[0].DrawMesh(rendering,self.Parts[bou].MC.Vertices,self.Parts[bou].MC.Normales,PoseBP, self.w.get(), self.color_tag)

        img = Image.fromarray(rendering, 'RGB')
        self.imgTk=ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
        if(self.skeleton_tag):
            self.DrawSkeleton2D(self.Pose)
        if(self.center_tag):
            self.DrawCenters2D(self.Pose)
        if(self.Sys_tag):
            self.DrawCenters2D(self.Pose)
            self.DrawSys2D(self.Pose)
        if(self.OBBox_tag):
            self.DrawOBBox2D(self.Pose)

        self.x_init = event.x
        self.y_init = event.y

    def DrawPoint2D(self,point,radius,color):
        """
        Draw a point in the image
        :param point: 2D coordinates
        :param radius: size of the point
        :param color: Color of the point
        :return: none
        """
        if point[0]>0 and point[1]>0:
            x1, y1 = (point[0] - radius), (point[1] - radius)
            x2, y2 = (point[0] + radius), (point[1] + radius)
        else:
            x1, y1 = (point[0]), (point[1])
            x2, y2 = (point[0]), (point[1])
        self.canvas.create_oval(x1, y1, x2, y2, fill=color)


    def DrawColors2D(self,RGBD,img):
        """
        Draw the color of each segmented part of the body
        :param RGBD: RGBD object
        :param img: previous image
        :return: unage
        """
        newImg = img.copy()
        Txy = RGBD.transCrop
        Txy[3] = min(Txy[3], img.shape[0])
        Txy[2] = min(Txy[2], img.shape[1])
        label = RGBD.labels
        for k in range(1,RGBD.bdyPart.shape[0]+1):
            color = RGBD.bdyColor[k-1]
            for i in range(Txy[1],Txy[3]):
                for j in range(Txy[0],Txy[2]):
                    if label[i][j]==k :
                        newImg[i,j] = color
                    else :
                        newImg[i,j] = newImg[i,j]
        return newImg


    def DrawSkeleton2D(self,Pose):
        """
        Sraw the Skeleton of a human and make connections between each part
        :param Pose: transformation
        :return
        """
        pos2D = self.pos2d[0][self.Index].astype(np.int16)-1
        pos = self.RGBD[0].GetProjPts2D_optimize(self.RGBD[0].Vtx[pos2D[:,1], pos2D[:,0]],Pose)

        for i in range(np.size(self.connection,0)):
            pt1 = (pos[self.connection[i,0]-1,0],pos[self.connection[i,0]-1,1])
            pt2 = (pos[self.connection[i,1]-1,0],pos[self.connection[i,1]-1,1])
            radius = 1
            color = "blue"
            self.DrawPoint2D(pt1,radius,color)
            self.DrawPoint2D(pt2,radius,color)
            self.canvas.create_line(pt1[0],pt1[1],pt2[0],pt2[1],fill="red")

    def DrawCenters2D(self,Pose,s=1):
        '''this function draw the center of each oriented coordinates system for each body part'''
        self.ctr2D = self.RGBD[0].GetProjPts2D_optimize(self.RGBD[0].ctr3D,Pose)
        for i in range(1, len(self.RGBD[0].ctr3D)):
            c = self.ctr2D[i]
            self.DrawPoint2D(c,2,"yellow")

    def DrawSys2D(self,Pose):
        '''this function draw the sys of oriented coordinates system for each body part'''
        # Compute the coordinates system of each body parts
        self.RGBD[0].GetNewSys(Pose,self.ctr2D,10)
        # Draw it
        for i in range(1,len(self.ctr2D)):
            # Get points to draw the coordinate system
            c = self.ctr2D[i]
            pt0 = self.RGBD[0].drawNewSys[i-1][0]
            pt1 = self.RGBD[0].drawNewSys[i-1][1]
            pt2 = self.RGBD[0].drawNewSys[i-1][2]
            # Draw the line of the coordinate system
            self.canvas.create_line(pt0[0],pt0[1],c[0],c[1],fill="red",width = 2)
            self.canvas.create_line(pt1[0],pt1[1],c[0],c[1],fill="green",width = 2)
            self.canvas.create_line(pt2[0],pt2[1],c[0],c[1],fill="blue",width = 2)

    def DrawOBBox2D(self,Pose):
        '''
        Draw in the canvas the Oriented Bounding Boxes (OBB) for each body part
        '''
        self.OBBcoords2D = []
        self.OBBcoords2D.append([0.,0.,0.])
        # for each body part
        for i in range(1,len(self.RGBD[0].coordsGbl)):
            # Get corners of OBB
            self.OBBcoords2D.append(self.RGBD[0].GetProjPts2D_optimize(self.RGBD[0].coordsGbl[i],Pose))
            pt = self.OBBcoords2D[i]
            # create lines of the boxes
            for j in range(3):
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+1][0],pt[j+1][1],fill="red",width =2)
                self.canvas.create_line(pt[j+4][0],pt[j+4][1],pt[j+5][0],pt[j+5][1],fill="red",width = 2)
                self.canvas.create_line(pt[j][0],pt[j][1],pt[j+4][0],pt[j+4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[0][0],pt[0][1],fill="red",width = 2)
            self.canvas.create_line(pt[7][0],pt[7][1],pt[4][0],pt[4][1],fill="red",width = 2)
            self.canvas.create_line(pt[3][0],pt[3][1],pt[7][0],pt[7][1],fill="red",width = 2)
            #draw points of the bounding boxes
            for j in range(8):
                self.DrawPoint2D(pt[j],2,"blue")

    ## Constructor function
    def __init__(self, path,  GPUManager, master=None):
        """
        Main function of the project
        :param path: path to search data
        :param GPUManager: GPU environment
        :param master: none
        """

        # Init
        self.root = master
        self.path = path
        self.GPUManager = GPUManager
        self.draw_bump = False # useless
        self.draw_spline = False # useless

        tk.Frame.__init__(self, master)
        self.pack()

        self.color_tag = 2
        # Calibration matrix
        calib_file = open(self.path + '/Calib.txt', 'r')
        calib_data = calib_file.readlines()
        self.Size = [int(calib_data[0]), int(calib_data[1])]
        self.intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                                   [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                                   [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)

        print self.intrinsic

        fact = 690

        TimeStart = time.time()

        #load data
        matfilename ='041_1027_01'
        mat = scipy.io.loadmat(path + '/' + matfilename + '.mat')
        lImages = mat['DepthImg']
        self.pos2d = mat['Pos2D']

        # use color image in Segmentation
        if 'Color2DepthImg' in mat:
            ColorImg = mat['Color2DepthImg']
        else:
            ColorImg = np.zeros((0))
        ColorImg = np.zeros((0))
        # initialization
        self.connectionMat = scipy.io.loadmat(path + '/SkeletonConnectionMap.mat')
        self.connection = self.connectionMat['SkeletonConnectionMap']
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        T_Pose = []
        PoseBP = np.zeros((15, 4, 4), dtype=np.float32)
        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)

        # number of images in the sequence. Start and End
        self.Index = 4
        nunImg = 8
        sImg = 1

        # Former Depth Image (i.e: i)
        self.RGBD = []
        # for each body compute the corresponding segmented image
        for bp in range(15):
            # add an RGBD Object in the list
            self.RGBD.append(RGBD.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', self.intrinsic, fact))
            # load data in the RGBD Object
            self.RGBD[bp].LoadMat(lImages,self.pos2d,self.connection, ColorImg)
            self.RGBD[bp].ReadFromMat(self.Index)
            # process depth image
            self.RGBD[bp].BilateralFilter(-1, 0.02, 3)
            # segmenting the body
            if bp == 0:
                self.RGBD[bp].RGBDSegmentation()
                self.RGBD[bp].depth_image *= (self.RGBD[bp].labels >0)
            else:
                self.RGBD[bp].depth_image *= (self.RGBD[0].labelList[bp] >0)
            # Compute vertex map and normal map
            self.RGBD[bp].Vmap_optimize()
            self.RGBD[bp].NMap_optimize()

        # show segmentation result (testing)

        img_label_temp =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        img_label_temp = self.DrawColors2D(self.RGBD[0],img_label_temp)
        img_label = img_label_temp.copy()
        img_label[:,:,0] = img_label_temp[:,:,2].copy()
        img_label[:,:,1] = img_label_temp[:,:,1].copy()
        img_label[:,:,2] = img_label_temp[:,:,0].copy()
        img_label[self.pos2d[0,self.Index][:,1].astype(np.int16), self.pos2d[0,self.Index][:,0].astype(np.int16), 1:3] = 110
        #cv2.imshow("depthimage", (self.RGBD[0].CroppedBox.astype(np.double))/7)
        #cv2.imshow("depthImage_threshold", (self.RGBD[0].BdyThresh()>0)*1.0)
        #print(str(self.Index) + " frame")
        #cv2.imshow("label", img_label)
        #cv2.waitKey()
        if(self.Index<10):
                imgstr = '00'+str(self.Index)
        elif(self.Index<100):
            imgstr = '0'+str(self.Index)
        else:
            imgstr = str(self.Index)
        cv2.imwrite('../segment/seg_'+ imgstr +'.png', img_label)


        # create the transform matrices that transform from local to global coordinate
        self.RGBD[0].myPCA()
        self.RGBD[0].BuildBB()
        self.RGBD[0].getWarpingPlanes()

        '''
        The first image is process differently from the other since it does not have any previous value.
        '''
        # Stock all Local to global Transform
        Tg = []
        Tg.append(Id4)
        # bp = 0 is the background (only black) No need to process it.
        for bp in range(1,self.RGBD[0].bdyPart.shape[0]+1):
            # Get the tranform matrix from the local coordinates system to the global system
            Tglo = self.RGBD[0].TransfoBB[bp]
            Tg.append(Tglo.astype(np.float32))

        # Sum of the number of vertices and faces of all body parts
        nb_verticesGlo = 0
        nb_facesGlo = 0
        # Number of body part (+1 since the counting starts from 1)
        bpstart = 1
        nbBdyPart = self.RGBD[0].bdyPart.shape[0]+1
        #Initialize stitcher object. It stitches the body parts
        StitchBdy = Stitcher.Stitch(nbBdyPart)
        boneTrans, boneSubTrans = StitchBdy.GetVBonesTrans(self.RGBD[0].skeVtx[0], self.RGBD[0].skeVtx[0])
        boneTr_all = boneTrans
        boneSubTr_all = boneSubTrans
        # Initialize Body parts
        Parts = []
        Parts.append(BdyPrt.BodyParts(self.GPUManager,self.RGBD[0],self.RGBD[0], Tg[0]))
        BPVtx = []
        BPVtx.append(np.array((0,0,0)))
        BPNml = []
        BPNml.append(np.array((0,0,0)))
        # Creating mesh of each body part
        for bp in range(bpstart,nbBdyPart):
            # get dual quaternion of bond's transformation
            boneMDQ, boneJDQ = StitchBdy.getJointInfo(bp, boneTrans, boneSubTrans)

            # create volume
            Parts.append(BdyPrt.BodyParts(self.GPUManager, self.RGBD[0], self.RGBD[bp], Tg[bp]))
            # Compute the 3D Model (TSDF + MC)
            Parts[bp].Model3D_init(bp, boneJDQ)

            # Update number of vertices and faces in the stitched mesh
            nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
            nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]

            #Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
            for i in range(4):
                for j in range(4):
                    PoseBP[bp][i][j] = Tg[bp][i][j]
            # Concatenate all the body parts for stitching purpose
            BPVtx.append(StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], Id4, Id4, 0, PoseBP[bp], bp))
            BPNml.append(StitchBdy.TransformNmls(Parts[bp].MC.Normales, Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], Id4, Id4, 0, PoseBP[bp], bp))
            if bp == bpstart  :
                StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp,1, self.RGBD[0])
                StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp,1, self.RGBD[0])
                StitchBdy.StitchedFaces = Parts[bp].MC.Faces
            else:
                StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, self.RGBD[0])
            # save vertex in global of each body part
            # Parts[1].MC.SaveToPlyExt("GBody"+str(self.Index)+"_"+str(bp)+".ply",Parts[bp].MC.nb_vertices[0],Parts[bp].MC.nb_faces[0],StitchBdy.TransformVtx(Parts[bp].MC.Vertices,self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp,1,self.RGBD[0]),Parts[bp].MC.Faces)

        # save with the number of the body part
        Parts[1].MC.SaveToPlyExt("wholeBody"+str(self.Index)+".ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces)
        #Parts[1].MC.SaveToPlyExt("skeleton"+str(self.Index)+".ply",21,0,self.RGBD[0].skeVtx[0, 0:21],[])
        # save the coordinate of the body part
        #for bp in range(bpstart, nbBdyPart):
            #Parts[1].MC.SaveBBToPlyExt("BB_"+str(self.Index)+"_"+str(bp)+".ply", self.RGBD[0].coordsGbl[bp], bp)
            #Parts[1].MC.SaveBBToPlyExt("BBL_"+str(self.Index)+"_"+str(bp)+".ply", self.RGBD[0].coordsL[bp], bp)

        # projection in 2d space to draw the 3D model(meshes) (testing)
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        bbrendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        rendering_oneVtx = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        rendering = self.RGBD[0].DrawMesh(rendering,StitchBdy.StitchedVertices,StitchBdy.StitchedNormales,Id4, 1, self.color_tag)
        cornernal =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        # show segmentation result
        img_label_temp = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        img_label_temp = self.DrawColors2D(self.RGBD[0], img_label_temp)
        img_label = img_label_temp.copy()
        img_label[:,:,0] = img_label_temp[:,:,2].copy()
        img_label[:,:,1] = img_label_temp[:,:,1].copy()
        img_label[:,:,2] = img_label_temp[:,:,0].copy()
        for i in range(3):
            for j in range(3):
                img_label[self.pos2d[0,self.Index][:,1].astype(np.int16)+i-1, self.pos2d[0,self.Index][:,0].astype(np.int16)+j-1,0] = 255
                img_label[self.pos2d[0,self.Index][:,1].astype(np.int16)+i-1, self.pos2d[0,self.Index][:,0].astype(np.int16)+j-1,1] = 220
                img_label[self.pos2d[0,self.Index][:,1].astype(np.int16)+i-1, self.pos2d[0,self.Index][:,0].astype(np.int16)+j-1,2] = 240
        img_label = img_label.astype(np.double)/255
        # show depth map result
        img_depthmap_temp =(self.RGBD[0].lImages[0][self.Index].astype(np.double)/7000*255).astype(np.uint8)
        img_depthmap = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        img_depthmap[:,:,0] = img_depthmap_temp.copy()
        img_depthmap[:,:,1] = img_depthmap_temp.copy()
        img_depthmap[:,:,2] = img_depthmap_temp.copy()
        img_skeleton = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        for i in range(3):
            for j in range(3):
                img_skeleton[self.pos2d[0,self.Index][:,1].astype(np.int16)+i-1, self.pos2d[0,self.Index][:,0].astype(np.int16)+j-1,0:3] = 1000
        # draw Boundingboxes
        for i in range(1,len(self.RGBD[0].coordsGbl)):
            # Get corners of OBB
            pt = self.RGBD[0].GetProjPts2D_optimize(self.RGBD[0].coordsGbl[i],Id4)
            pt[:,0] = np.maximum(0,np.minimum(pt[:,0], self.Size[1]-1))
            pt[:,1] = np.maximum(0,np.minimum(pt[:,1], self.Size[0]-1))
            # create point of the boxes
            bbrendering[pt[:,1], pt[:,0],0] = 100
            bbrendering[pt[:,1], pt[:,0],1] = 230
            bbrendering[pt[:,1], pt[:,0],2] = 230
            if i==10:
                for j in range(7):
                    rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+1][1],pt[j+1][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+9][1],pt[j+9][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[j+9][1],pt[j+9][0],pt[j+10][1],pt[j+10][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                rr,cc,val = line_aa(pt[0][1],pt[0][0],pt[8][1],pt[8][0])
                rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                bbrendering[rr,cc, 0] = 100
                bbrendering[rr,cc, 1] = 230
                bbrendering[rr,cc, 2] = 250
                rr,cc,val = line_aa(pt[9][1],pt[9][0],pt[17][1],pt[17][0])
                rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                bbrendering[rr,cc, 0] = 100
                bbrendering[rr,cc, 1] = 230
                bbrendering[rr,cc, 2] = 250
            else:
                # create lines of the boxes
                for j in range(3):
                    rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+1][1],pt[j+1][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[j+4][1],pt[j+4][0],pt[j+5][1],pt[j+5][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+4][1],pt[j+4][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                rr,cc,val = line_aa(pt[3][1],pt[3][0],pt[0][1],pt[0][0])
                rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                bbrendering[rr,cc, 0] = 100
                bbrendering[rr,cc, 1] = 230
                bbrendering[rr,cc, 2] = 250
                rr,cc,val = line_aa(pt[7][1],pt[7][0],pt[4][1],pt[4][0])
                rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                bbrendering[rr,cc, 0] = 100
                bbrendering[rr,cc, 1] = 230
                bbrendering[rr,cc, 2] = 250
                rr,cc,val = line_aa(pt[3][1],pt[3][0],pt[7][1],pt[7][0])
                rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                bbrendering[rr,cc, 0] = 100
                bbrendering[rr,cc, 1] = 230
                bbrendering[rr,cc, 2] = 250
        # mix
        result_stack = np.concatenate((rendering*0.0020+img_depthmap*0.0020+rendering_oneVtx*0.0025, np.ones((self.Size[0],1,3), dtype = np.uint8), bbrendering*0.001+img_depthmap*0.0020+img_skeleton*0.001+cornernal*0.003), axis=1)
        #result_stack = np.concatenate((result_stack, np.ones((self.Size[0],1,3), dtype = np.uint8)*255, img_label), axis=1)
        print ("frame"+str(self.Index))
        cv2.imshow("BB", result_stack)
        cv2.waitKey(1)
        if(self.Index<10):
                imgstr = '00'+str(self.Index)
        elif(self.Index<100):
            imgstr = '0'+str(self.Index)
        else:
            imgstr = str(self.Index)
        cv2.imwrite('../boundingboxes/bb_'+ imgstr +'.png', bbrendering)
        cv2.imwrite('../normal/nml_'+ imgstr +'.png', rendering)

        #as prev RGBD
        newRGBD = self.RGBD

        # initialize tracker for camera pose
        Tracker = TrackManager.Tracker(0.001, 0.5, 1, [10])
        formerIdx = self.Index
        for bp in range(nbBdyPart+1):
            T_Pose.append(Id4)

        for imgk in range(self.Index+sImg,nunImg, sImg):
            #Time counting
            start = time.time()

            '''
            New Image
            '''
            # save pre RGBD
            preRGBD = newRGBD

            # Current Depth Image (i.e: i+1)
            newRGBD = []
            Tbb_s = []
            Tbb_icp = []
            Tbb_s.append(Id4)
            Tbb_icp.append(Id4)

            # separate  each body parts of the image into different object -> each object have just the body parts in its depth image
            for bp in range(nbBdyPart):
                newRGBD.append(RGBD.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', self.intrinsic, fact))
                newRGBD[bp].LoadMat(lImages,self.pos2d,self.connection, ColorImg)
                # Get new current image
                newRGBD[bp].ReadFromMat(imgk)
                newRGBD[bp].BilateralFilter(-1, 0.02, 3)
                # segmenting the body/select the body part
                if bp == 0:
                    newRGBD[bp].RGBDSegmentation()
                    newRGBD[bp].depth_image *= (newRGBD[bp].labels > 0)
                else:
                    newRGBD[bp].depth_image *= (newRGBD[0].labelList[bp] >0)
                # Vertex and Normal map
                newRGBD[bp].Vmap_optimize()
                newRGBD[bp].NMap_optimize()

            # show segmentation result (testing)
            #'''
            img_label_temp =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            img_label_temp = self.DrawColors2D(newRGBD[0],img_label_temp)
            img_label = img_label_temp.copy()
            img_label[:,:,0] = img_label_temp[:,:,2]
            img_label[:,:,1] = img_label_temp[:,:,1]
            img_label[:,:,2] = img_label_temp[:,:,0]
            img_label[self.pos2d[0,imgk][:,1].astype(np.int16), self.pos2d[0,imgk][:,0].astype(np.int16), 1:3] = 110
            #print(str(imgk)+" frame")
            #cv2.imshow("depthimage", (newRGBD[0].CroppedBox.astype(np.double))/7)
            #cv2.imshow("depthImage_threshold", (newRGBD[0].BdyThresh()>0)*1.0)
            #cv2.imshow("label", img_label)
            #cv2.waitKey(1)
            if(imgk<10):
                imgstr = '00'+str(imgk)
            elif(imgk<100):
                imgstr = '0'+str(imgk)
            else:
                imgstr = str(imgk)
            cv2.imwrite('../segment/seg_'+ imgstr +'.png', img_label)
            #'''

            # creating mesh of whole body (testing)
            tempVtx = np.zeros((sum(sum(newRGBD[0].Vtx[:,:,2]>0)), 3))
            tempt = 0
            for i in range(newRGBD[0].Vtx.shape[0]):
                for j in range(newRGBD[0].Vtx.shape[1]):
                    if newRGBD[0].Vtx[i,j,2]>0:
                        tempVtx[tempt,:] = newRGBD[0].Vtx[i,j,:]
                        tempt+=1
            rendering_originmesh = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            rendering_originmesh = self.RGBD[0].DrawMesh(rendering_originmesh,tempVtx,tempVtx,Id4, 1, 1)
            Parts[bp].MC.SaveToPlyExt("wholeBody"+str(imgk)+"_ori.ply",tempt,0,StitchBdy.TransformVtx(tempVtx,self.RGBD[0].coordsGbl[bp],self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], Id4,Id4, 0, Id4, bp),[],0)


            # create the transform matrix from local to global coordinate
            newRGBD[0].myPCA()
            newRGBD[0].BuildBB()

            # Tracking and get transform the bounding-boxes into current image
            boneTrans, boneSubTrans = StitchBdy.GetVBonesTrans(newRGBD[0].skeVtx[0], preRGBD[0].skeVtx[0])
            for binx in range(20): # first to previous frame + previous to current frame
                boneTr_all[binx] = np.dot(boneTrans[binx], boneTr_all[binx])
                boneTrans[binx] = np.dot(np.identity(4), boneTr_all[binx])
                boneSubTrans[binx] = np.dot(np.identity(4), boneSubTr_all[binx])
            boneTrans = Tracker.RegisterBoneTr( boneTrans, BPVtx, BPNml, tempVtx, newRGBD[0].depth_image, newRGBD[0].intrinsic, self.RGBD[0].planesF)
            newRGBD[0].coordsGbl, newRGBD[0].BBTrans = StitchBdy.TransfoBBcorners(self.RGBD[0].skeVtx[0], self.RGBD[0].coordsGbl, self.RGBD[0].BBTrans)

            # Sum of the number of vertices and faces of all body parts
            nb_verticesGlo = 0
            nb_facesGlo = 0
            #Initiate stitcher object
            StitchBdy = Stitcher.Stitch(nbBdyPart)

            # projection in 2d space to draw the 3D model(meshes) (testing)
            rendering = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            bbrendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            rendering_oneVtx = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            cornernal =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)

            BPVtx = []
            BPVtx.append(np.array((0,0,0)))
            BPNml = []
            BPNml.append(np.array((0,0,0)))
            # Updating mesh of each body part
            for bp in range(1,nbBdyPart):
                print "bp: " + str(bp)

                # get bp's joint information
                boneMDQ, boneJDQ = StitchBdy.getJointInfo(bp, boneTrans, boneSubTrans)

                # TSDF Fusion of the body part
                Parts[bp].TSDFManager.FuseRGBD_GPU(newRGBD[bp], boneMDQ[0], boneJDQ[0])

                # Create Mesh
                Parts[bp].MC = My_MC.My_MarchingCube(Parts[bp].TSDFManager.Size, Parts[bp].TSDFManager.res, 0.0, self.GPUManager)
                # Mesh rendering
                Parts[bp].MC.runGPU(Parts[bp].TSDFManager.TSDFGPU)

                # Update number of vertices and faces in the stitched mesh
                nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
                nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]

                # Stitch all the body parts
                BPVtx.append(StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], Id4, Id4, 0, PoseBP[bp], bp))
                BPNml.append(StitchBdy.TransformNmls(Parts[bp].MC.Normales, Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], Id4, Id4, 0, PoseBP[bp], bp))
                if bp ==1 :
                    StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0])
                    StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0])
                    StitchBdy.StitchedFaces = Parts[bp].MC.Faces
                else:
                    StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, self.RGBD[0])

                # output mesh in global of each body part (testing)
                #Parts[bp].MC.SaveToPlyExt("GBody"+str(imgk)+"_"+str(bp)+".ply",Parts[bp].MC.nb_vertices[0],Parts[bp].MC.nb_faces[0],StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0]),Parts[bp].MC.Faces,0)
                #Parts[bp].MC.SaveToPly("body" + str(imgk)+"_" +str(bp) + ".ply")

            # projection in 2d space to draw the 3D model(meshes) (testing)
            rendering = self.RGBD[0].DrawMesh(rendering,StitchBdy.StitchedVertices,StitchBdy.StitchedNormales,Id4, 1, self.color_tag)

            formerIdx = imgk
            time_lapsed = time.time() - start
            print "number %d finished : %f" %(imgk,time_lapsed)

            # save with the number of the body part
            imgkStr = str(imgk)
            Parts[bp].MC.SaveToPlyExt("wholeBody"+imgkStr+".ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces,0)
            #Parts[1].MC.SaveToPlyExt("skeleton"+imgkStr+".ply",21,0,newRGBD[0].skeVtx[0, 0:21],[])
            # save the coordinate of the body part
            #for bp in range(bpstart, nbBdyPart):
                #Parts[1].MC.SaveBBToPlyExt("BB_"+imgkStr+"_"+str(bp)+".ply", StitchBdy.TransformVtx(newRGBD[0].coordsGbl[bp],self.RGBD[0].coordsGbl[bp],self.RGBD[0].coordsGbl[bp], self.RGBD[0].BBTrans[bp], Id4, Id4, 0,Id4, bp), bp)

            # save model in first frame space
            nb_verticesGlo = 0
            nb_facesGlo = 0
            StitchBdy = Stitcher.Stitch(nbBdyPart)
            boneTrans, boneSubTrans = StitchBdy.GetVBonesTrans(self.RGBD[0].skeVtx[0], self.RGBD[0].skeVtx[0])
            for bp in range(1,nbBdyPart):
                boneMDQ, boneJDQ = StitchBdy.getJointInfo(bp, boneTrans, boneSubTrans)
                nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
                nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]
                if bp ==1 :
                    StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0])
                    StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0])
                    StitchBdy.StitchedFaces = Parts[bp].MC.Faces
                else:
                    StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, self.RGBD[0])
            Parts[bp].MC.SaveToPlyExt("wholeBody"+str(imgkStr)+"F.ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces,0)

            # projection in 2d space to draw the 3D model(meshes) (testing)
            # show segmentation result
            img_label_temp = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            img_label_temp = self.DrawColors2D(newRGBD[0], img_label_temp)
            img_label = img_label_temp.copy()
            img_label[:,:,0] = img_label_temp[:,:,2].copy()
            img_label[:,:,1] = img_label_temp[:,:,1].copy()
            img_label[:,:,2] = img_label_temp[:,:,0].copy()
            for i in range(3):
                for j in range(3):
                    img_label[self.pos2d[0,imgk][:,1].astype(np.int16)+i-1, self.pos2d[0,imgk][:,0].astype(np.int16)+j-1,0] = 255
                    img_label[self.pos2d[0,imgk][:,1].astype(np.int16)+i-1, self.pos2d[0,imgk][:,0].astype(np.int16)+j-1,1] = 220
                    img_label[self.pos2d[0,imgk][:,1].astype(np.int16)+i-1, self.pos2d[0,imgk][:,0].astype(np.int16)+j-1,2] = 240
            img_label = img_label.astype(np.double)/255
            # show depth map result
            img_depthmap = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            img_depthmap_temp =(self.RGBD[0].lImages[0][imgk].astype(np.double)/7000*255).astype(np.uint8)
            img_depthmap[:,:,0] = img_depthmap_temp
            img_depthmap[:,:,1] = img_depthmap_temp
            img_depthmap[:,:,2] = img_depthmap_temp
            img_skeleton = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            for i in range(3):
                for j in range(3):
                    img_skeleton[self.pos2d[0,imgk][:,1].astype(np.int16)+i-1, self.pos2d[0,imgk][:,0].astype(np.int16)+j-1,1:3] = 1000
            # draw Boundingboxes
            rendering_oneVtx = np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
            for i in range(1,len(self.RGBD[0].coordsGbl)):

                # Get corners of OBB
                pt = self.RGBD[0].GetProjPts2D_optimize(newRGBD[0].coordsGbl[i], Id4)
                pt[:,0] = np.maximum(0,np.minimum(pt[:,0], self.Size[1]-1))
                pt[:,1] = np.maximum(0,np.minimum(pt[:,1], self.Size[0]-1))
                # create point of the boxes
                bbrendering[pt[:,1], pt[:,0],0] = 100
                bbrendering[pt[:,1], pt[:,0],1] = 230
                bbrendering[pt[:,1], pt[:,0],2] = 230
                if i==10:
                    for j in range(7):
                        rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+1][1],pt[j+1][0])
                        rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                        cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                        bbrendering[rr,cc, 0] = 100
                        bbrendering[rr,cc, 1] = 230
                        bbrendering[rr,cc, 2] = 250
                        rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+9][1],pt[j+9][0])
                        rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                        cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                        bbrendering[rr,cc, 0] = 100
                        bbrendering[rr,cc, 1] = 230
                        bbrendering[rr,cc, 2] = 250
                        rr,cc,val = line_aa(pt[j+9][1],pt[j+9][0],pt[j+10][1],pt[j+10][0])
                        rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                        cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                        bbrendering[rr,cc, 0] = 100
                        bbrendering[rr,cc, 1] = 230
                        bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[0][1],pt[0][0],pt[8][1],pt[8][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[9][1],pt[9][0],pt[17][1],pt[17][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                else:
                    # create lines of the boxes
                    for j in range(3):
                        rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+1][1],pt[j+1][0])
                        rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                        cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                        bbrendering[rr,cc, 0] = 100
                        bbrendering[rr,cc, 1] = 230
                        bbrendering[rr,cc, 2] = 250
                        rr,cc,val = line_aa(pt[j+4][1],pt[j+4][0],pt[j+5][1],pt[j+5][0])
                        rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                        cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                        bbrendering[rr,cc, 0] = 100
                        bbrendering[rr,cc, 1] = 230
                        bbrendering[rr,cc, 2] = 250
                        rr,cc,val = line_aa(pt[j][1],pt[j][0],pt[j+4][1],pt[j+4][0])
                        rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                        cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                        bbrendering[rr,cc, 0] = 100
                        bbrendering[rr,cc, 1] = 230
                        bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[3][1],pt[3][0],pt[0][1],pt[0][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[7][1],pt[7][0],pt[4][1],pt[4][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250
                    rr,cc,val = line_aa(pt[3][1],pt[3][0],pt[7][1],pt[7][0])
                    rr = np.maximum(0,np.minimum(rr, self.Size[0]-1))
                    cc = np.maximum(0,np.minimum(cc, self.Size[1]-1))
                    bbrendering[rr,cc, 0] = 100
                    bbrendering[rr,cc, 1] = 230
                    bbrendering[rr,cc, 2] = 250

            # mix
            result_stack = np.concatenate((rendering*0.0020+img_depthmap*0.0020+rendering_oneVtx*0.0025, np.ones((self.Size[0],1,3), dtype = np.uint8), bbrendering*0.001+img_depthmap*0.0020+img_skeleton*0.001+cornernal*0.003), axis=1)
            #result_stack = np.concatenate((result_stack, np.ones((self.Size[0],1,3), dtype = np.uint8)*255, img_label), axis=1)
            print ("frame"+imgkStr)
            cv2.imshow("BB", result_stack)
            cv2.waitKey(1)
            if(imgk<10):
                imgstr = '00'+str(imgk)
            elif(imgk<100):
                imgstr = '0'+str(imgk)
            else:
                imgstr = str(imgk)
            cv2.imwrite('../boundingboxes/bb_'+ imgstr +'.png', bbrendering)
            cv2.imwrite('../normal/nml_'+ imgstr +'.png', rendering)

        TimeStart_Lapsed = time.time() - TimeStart
        print "total time: %f" %(TimeStart_Lapsed)

        # save final model in first frame
        nb_verticesGlo = 0
        nb_facesGlo = 0
        StitchBdy = Stitcher.Stitch(nbBdyPart)
        boneTrans, boneSubTrans = StitchBdy.GetVBonesTrans(self.RGBD[0].skeVtx[0], self.RGBD[0].skeVtx[0])
        for bp in range(1,nbBdyPart):
            boneMDQ, boneJDQ = StitchBdy.getJointInfo(bp, boneTrans, boneSubTrans)
            nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
            nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]
            if bp ==1 :
                StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0])
                StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,Parts[bp].MC.Vertices, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, 1, self.RGBD[0])
                StitchBdy.StitchedFaces = Parts[bp].MC.Faces
            else:
                StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces, self.RGBD[0].coordsGbl[bp], newRGBD[0].coordsGbl[bp], newRGBD[0].BBTrans[bp], boneMDQ, boneJDQ, self.RGBD[0].planesF[bp], PoseBP[bp], bp, self.RGBD[0])
        Parts[bp].MC.SaveToPlyExt("wholeBody"+str(self.Index)+"F.ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces,0)

        # show UI result
        # projection in 2d space to draw it
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)

        # Projection for each body parts done separately
        for bp in range(bpstart,nbBdyPart):
            bou = bp
            for i in range(4):
                for j in range(4):
                    PoseBP[bp][i][j] = Parts[bou].Tlg[i][j]
            rendering = self.RGBD[0].DrawMesh(rendering,Parts[bou].MC.Vertices,Parts[bou].MC.Normales,PoseBP[bp], 1, self.color_tag)

        # drawing tag
        self.skeleton_tag = 0
        self.center_tag = 0
        self.Sys_tag = 0
        self.OBBox_tag = 0
        self.first_tag = 0
        # save fused Vertices and Normals
        self.Parts = Parts

        # 3D reconstruction of the whole image
        self.canvas = tk.Canvas(self, bg="black", height=self.Size[0], width=self.Size[1])
        self.canvas.pack()
        #rendering = self.DrawColors2D(self.RGBD[0],rendering,self.Pose)
        img = Image.fromarray(rendering, 'RGB')
        self.imgTk=ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.imgTk)
        if(self.skeleton_tag):
            self.DrawSkeleton2D(self.Pose)
        if(self.center_tag):
            self.DrawCenters2D(self.Pose)
        if(self.Sys_tag):
            self.DrawSys2D(self.Pose)
        if(self.OBBox_tag):
            self.DrawOBBox2D(self.Pose)

        #enable keyboard and mouse monitoring
        self.root.bind("<Key>", self.key)
        self.root.bind("<Button-1>", self.mouse_press)
        self.root.bind("<ButtonRelease-1>", self.mouse_release)
        self.root.bind("<B1-Motion>", self.mouse_motion)

        self.w = tk.Scale(master, from_=1, to=10, orient=tk.HORIZONTAL)
        self.w.pack()


