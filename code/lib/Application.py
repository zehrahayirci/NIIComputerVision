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
        :param RGBD:
        :param img:
        :param Pose:
        :return:
        Adding an argument pose would enable to follow the transformation
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
        :param Pose:
        :return:
                Adding an argument pose would enable to follow the transformation
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
        
        self.color_tag = 1
        # Calibration matrix
        calib_file = open(self.path + '/Calib.txt', 'r')
        calib_data = calib_file.readlines()
        self.Size = [int(calib_data[0]), int(calib_data[1])]
        self.intrinsic = np.array([[float(calib_data[2]), float(calib_data[3]), float(calib_data[4])], \
                                   [float(calib_data[5]), float(calib_data[6]), float(calib_data[7])], \
                                   [float(calib_data[8]), float(calib_data[9]), float(calib_data[10])]], dtype = np.float32)
    
        print self.intrinsic

        fact = 650

        TimeStart = time.time()

        #load data
        path2 = 'C:/Users/nii-user/Desktop/sylvia/Kinect_dataset'
        matfilename = '031_0915_01'
        mat = scipy.io.loadmat(path2 + '/' + matfilename + '.mat')
        #mat = scipy.io.loadmat(path + '/String4b.mat')
        lImages = mat['DepthImg']
        self.pos2d = mat['Pos2D']

        self.connectionMat = scipy.io.loadmat(path + '/SkeletonConnectionMap.mat')
        self.connection = self.connectionMat['SkeletonConnectionMap']
        self.Pose = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        T_Pose = []
        PoseBP = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        Id4 = np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]], dtype = np.float32)
        
        # number of images in the sequence. Start and End
        self.Index = 3
        nunImg = 14
        sImg = 1

        # Former Depth Image (i.e: i)
        self.RGBD = []
        # for each body compute the corresponding segmented image
        for bp in range(15):
            # add an RGBD Object in the list
            self.RGBD.append(RGBD.RGBD(path + '/Depth.tiff', path + '/RGB.tiff', self.intrinsic, fact))
            # load data in the RGBD Object
            self.RGBD[bp].LoadMat(lImages,self.pos2d,self.connection)
            self.RGBD[bp].ReadFromMat(self.Index)
            # process depth image
            self.RGBD[bp].BilateralFilter(-1, 0.02, 3)
            # segmenting the body
            if bp == 0:
                self.RGBD[bp].RGBDSegmentation()
                self.RGBD[bp].depth_image *= (self.RGBD[bp].labels >0)
            else:
                self.RGBD[bp].depth_image *= (self.RGBD[0].labels == bp)
            # Compute vertex map and normal map
            self.RGBD[bp].Vmap_optimize()   
            self.RGBD[bp].NMap_optimize()

        # create the transform matrices that transform from local to global coordinate
        self.RGBD[0].myPCA()

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
        # Initialize Body parts
        Parts = []
        Parts.append(BdyPrt.BodyParts(self.GPUManager,self.RGBD[0],self.RGBD[0], Tg[0]))
        # Creating mesh of each body part
        for bp in range(bpstart,nbBdyPart):
            Parts.append(BdyPrt.BodyParts(self.GPUManager, self.RGBD[0], self.RGBD[bp], Tg[bp]))
            # Compute the 3D Model (TSDF + MC)
            Parts[bp].Model3D_init(bp)

            # Update number of vertices and faces in the stitched mesh
            nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
            nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]

            #Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
            for i in range(4):
                for j in range(4):
                    PoseBP[i][j] = Tg[bp][i][j]
            # Concatenate all the body parts for stitching purpose
            if bp == bpstart :
                StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices,PoseBP,1)
                StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,PoseBP,1)
                StitchBdy.StitchedFaces = Parts[bp].MC.Faces
            else:
                StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces,PoseBP)

        # save with the number of the body part
        Parts[1].MC.SaveToPlyExt("wholeBody.ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces)
        
        # projection in 2d space to draw the 3D model
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)
        for bp in range(bpstart,nbBdyPart):
            bou = bp
            for i in range(4):
                for j in range(4):
                    PoseBP[i][j] = Parts[bou].Tlg[i][j]
            rendering = self.RGBD[0].DrawMesh(rendering,Parts[bou].MC.Vertices,Parts[bou].MC.Normales,PoseBP, 1, self.color_tag)
        # show segmentation result
        img_label_temp =(self.RGBD[0].lImages[0][self.Index].astype(np.double)/7000*255).astype(np.uint8)
        img_label = rendering.copy()
        img_label[:,:,0] = img_label_temp.copy()
        img_label[:,:,1] = img_label_temp.copy()
        img_label[:,:,2] = img_label_temp.copy()
        img_label[self.pos2d[0,self.Index][:,1].astype(np.int16), self.pos2d[0,self.Index][:,0].astype(np.int16),1:3] = 1000
        # mix
        result_stack = np.concatenate((rendering*0.0025+img_label*0.0025, np.ones((self.Size[0],1,3), dtype = np.uint8)*255, img_label*0.005), axis=1)
        print ("frame"+str(self.Index))
        cv2.imshow("Tracking", result_stack)
        cv2.waitKey(1)
        if(self.Index<10):
                imgstr = '00'+str(self.Index)
        elif(self.Index<100):
            imgstr = '0'+str(self.Index)
        else:
            imgstr = str(self.Index)
        cv2.imwrite('C:/Users/nii-user/Desktop/sylvia/results/tracking/' + matfilename + '/track_'+ imgstr +'.png', result_stack*255)

        #as prev RGBD
        newRGBD = self.RGBD
        
        #"""
        # initialize tracker for camera pose
        Tracker = TrackManager.Tracker(0.001, 0.5, 1, [10])
        formerIdx = self.Index

        for bp in range(nbBdyPart+1):
            T_Pose.append(Id4)
            
        for imgk in range(self.Index+1,nunImg, sImg):
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
                newRGBD[bp].LoadMat(lImages,self.pos2d,self.connection)              
                # Get new current image
                newRGBD[bp].ReadFromMat(imgk) 
                newRGBD[bp].BilateralFilter(-1, 0.02, 3)
                # segmenting the body/select the body part
                if bp == 0:
                    newRGBD[bp].RGBDSegmentation()
                    newRGBD[bp].depth_image *= (newRGBD[bp].labels > 0) 
                else:
                    newRGBD[bp].depth_image *= (newRGBD[0].labels == bp)
                # Vertex and Normal map
                newRGBD[bp].Vmap_optimize()   
                newRGBD[bp].NMap_optimize()        
            # create the transform matrix from local to global coordinate
            newRGBD[0].myPCA()

            # Transform the stitch body in the current image (alignment current image mesh) 
            # New pose estimation
            NewPose = Tracker.RegisterRGBDMesh_optimize(newRGBD[0],newRGBD[0].skeVtx[0],preRGBD[0].skeVtx[0],StitchBdy.StitchedVertices,StitchBdy.StitchedNormales, Id4)
            # Sum of the number of vertices and faces of all body parts
            nb_verticesGlo = 0
            nb_facesGlo = 0
            
            #Initiate stitcher object 
            StitchBdy = Stitcher.Stitch(nbBdyPart)      
            
            # projection in 2d space to draw the 3D model
            rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)

            
            # Updating mesh of each body part
            for bp in range(1,nbBdyPart):
                # Transform in the current image
                #Skeleton tracking
                print "BP: ", bp
                Tbb_s.append(StitchBdy.GetBBTransfo(self.pos2d, imgk, formerIdx, self.RGBD[0], preRGBD[0], newRGBD[0], bp, NewPose))
                Tbb_icp.append(Tracker.RegisterRGBDMesh_optimize(newRGBD[bp],newRGBD[0].skeVtx[bp],preRGBD[0].skeVtx[bp],StitchBdy.TransformVtx(Parts[bp].MC.Vertices,np.dot(T_Pose[bp],Tg[bp]),1),StitchBdy.TransformNmls(Parts[bp].MC.Normales,np.dot(T_Pose[bp],Tg[bp]),1), np.dot(Tbb_s[bp], NewPose)) )
                #T_Pose[bp] = np.dot(NewPose, T_Pose[bp])
                #T_Pose[bp] = np.dot(Tbb_s[bp], T_Pose[bp])
                T_Pose[bp] = np.dot(Tbb_icp[bp], T_Pose[bp])
                
                #update transform matrix with camera pose & local pose
                Tg_new = np.dot(T_Pose[bp],Tg[bp])
                
                # Put the Global transfo in PoseBP so that the dtype entered in the GPU is correct
                for i in range(4):
                    for j in range(4):
                        PoseBP[i][j] = Tg_new[i][j]#Tg[bp][i][j]#

                # projection in 2d space to draw the 3D model
                rendering = self.RGBD[0].DrawMesh(rendering,Parts[bp].MC.Vertices,Parts[bp].MC.Normales,PoseBP, 1, self.color_tag)

                # TSDF Fusion of the body part
                # Parts[bp].TSDFManager.FuseRGBD_GPU(newRGBD[bp], PoseBP)

                # Create Mesh
                # Parts[bp].MC = My_MC.My_MarchingCube(Parts[bp].TSDFManager.Size, Parts[bp].TSDFManager.res, 0.0, self.GPUManager)
                # Mesh rendering
                # Parts[bp].MC.runGPU(Parts[bp].TSDFManager.TSDFGPU)
    
                # Update number of vertices and faces in the stitched mesh
                nb_verticesGlo = nb_verticesGlo + Parts[bp].MC.nb_vertices[0]
                nb_facesGlo = nb_facesGlo +Parts[bp].MC.nb_faces[0]
                
                # Stitch all the body parts
                if bp ==1 :
                    StitchBdy.StitchedVertices = StitchBdy.TransformVtx(Parts[bp].MC.Vertices,PoseBP,1)
                    StitchBdy.StitchedNormales = StitchBdy.TransformNmls(Parts[bp].MC.Normales,PoseBP,1)
                    StitchBdy.StitchedFaces = Parts[bp].MC.Faces
                else:
                    StitchBdy.NaiveStitch(Parts[bp].MC.Vertices,Parts[bp].MC.Normales,Parts[bp].MC.Faces,PoseBP)
            
            formerIdx = imgk
            time_lapsed = time.time() - start
            print "number %d finished : %f" %(imgk,time_lapsed)

            # save with the number of the body part
            imgkStr = str(imgk)
            Parts[bp].MC.SaveToPlyExt("wholeBody"+imgkStr+".ply",nb_verticesGlo,nb_facesGlo,StitchBdy.StitchedVertices,StitchBdy.StitchedFaces,0)

            # projection in 2d space to draw the 3D model
            # show segmentation result
            img_label_temp =(self.RGBD[0].lImages[0][imgk].astype(np.double)/7000*255).astype(np.uint8)
            img_label = rendering.copy()
            img_label[:,:,0] = img_label_temp.copy()
            img_label[:,:,1] = img_label_temp.copy()
            img_label[:,:,2] = img_label_temp.copy()
            img_label[self.pos2d[0,imgk][:,1].astype(np.int16), self.pos2d[0,imgk][:,0].astype(np.int16),1:3] = 1000
            # mix
            result_stack = np.concatenate((rendering*0.0025+img_label*0.0025, np.ones((self.Size[0],1,3), dtype = np.uint8)*255, img_label*0.005), axis=1)
            print ("frame"+imgkStr)
            cv2.imshow("Tracking", result_stack)
            cv2.waitKey(1)
            if(imgk<10):
                imgstr = '00'+str(imgk)
            elif(imgk<100):
                imgstr = '0'+str(imgk)
            else:
                imgstr = str(imgk)
            cv2.imwrite('C:/Users/nii-user/Desktop/sylvia/results/tracking/' + matfilename + '/track_' + imgstr + '.png', result_stack*255)

        TimeStart_Lapsed = time.time() - TimeStart
        print "total time: %f" %(TimeStart_Lapsed)
        #"""

        # projection in 2d space to draw it
        rendering =np.zeros((self.Size[0], self.Size[1], 3), dtype = np.uint8)

        # Projection for each body parts done separately
        for bp in range(bpstart,nbBdyPart):
            bou = bp
            for i in range(4):
                for j in range(4):
                    PoseBP[i][j] = Parts[bou].Tlg[i][j]
            rendering = self.RGBD[0].DrawMesh(rendering,Parts[bou].MC.Vertices,Parts[bou].MC.Normales,PoseBP, 1, self.color_tag)

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
        exit()

