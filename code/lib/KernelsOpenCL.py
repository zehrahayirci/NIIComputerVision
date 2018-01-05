#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 11:21:40 2017

@author: diegothomas
"""
Kernel_Test = """
__kernel void Test(__global float *TSDF) {

        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        int z = get_global_id(2); /*depth*/
        TSDF[x + 512*y + 512*512*z] = 1.0f;
}
"""
#__global float *prevTSDF, __global float *Weight
#__read_only image2d_t VMap
Kernel_FuseTSDF = """
__kernel void FuseTSDF(__global short int *TSDF,  __global float *Depth, __constant float *Param, __constant int *Dim,
                           __constant float *Pose, 
                           __constant float *coordC,  __constant float *coordNew, const short int bp,
                           __constant float *calib, const int n_row, const int m_col, __global short int *Weight) {
        //const sampler_t smp =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_NONE | CLK_FILTER_NEAREST;

        const float nu = 0.05f;

            
        float4 pt;
        float4 ctr;
        float4 pt_T;
        float4 ctr_T;
        int2 pix;        
        
        int x = get_global_id(0); /*height*/
        int y = get_global_id(1); /*width*/
        pt.x = ((float)(x)-Param[0])/Param[1];
        pt.y = ((float)(y)-Param[2])/Param[3];
        float x_T =  Pose[0]*pt.x + Pose[1]*pt.y + Pose[3];
        float y_T =  Pose[4]*pt.x + Pose[5]*pt.y + Pose[7];
        float z_T =  Pose[8]*pt.x + Pose[9]*pt.y + Pose[11];
            
        
        float convVal = 32767.0f;
        int z ;
        for ( z = 0; z < Dim[2]; z++) { /*depth*/
            // On the GPU all pixel are stocked in a 1D array
            int idx = z + Dim[2]*y + Dim[2]*Dim[1]*x;

            // Transform voxel coordinates into 3D point coordinates
            // Param = [c_x, dim_x, c_y, dim_y, c_z, dim_z]
            pt.z = ((float)(z)-Param[4])/Param[5];          
            
            // Transfom the voxel into the Image coordinate space
            //transform form local to global
            pt_T.x = x_T + Pose[2]*pt.z; //Pose is column major
            pt_T.y = y_T + Pose[6]*pt.z;
            pt_T.z = z_T + Pose[10]*pt.z;
            //transform from first frame to current frame according interploation
            if(bp!=10){
                float dis1[2], dis3[2];
                int wlist[6][2];
                dis1[0] = fabs(coordC[0*3+0]-coordC[1*3+0]);
                dis1[1] = fabs(coordC[0*3+1]-coordC[1*3+1]);
                dis3[0] = fabs(coordC[0*3+0]-coordC[3*3+0]);
                dis3[1] = fabs(coordC[0*3+1]-coordC[3*3+1]);
                if(dis1[1]==0 || dis3[0]==0 || dis1[0]/dis1[1]>dis3[0]/dis3[1]){
                    wlist[0][0] = 0;
                    wlist[0][1] = 1;
                    wlist[1][0] = 3;
                    wlist[1][1] = 2;
                    wlist[2][0] = 4;
                    wlist[2][1] = 5;
                    wlist[3][0] = 7;
                    wlist[3][1] = 6;
                }
                else{
                    wlist[0][0] = 0;
                    wlist[0][1] = 3;
                    wlist[1][0] = 1;
                    wlist[1][1] = 2;
                    wlist[2][0] = 4;
                    wlist[2][1] = 7;
                    wlist[3][0] = 5;
                    wlist[3][1] = 6;
                }
                if(wlist[1][0]==1 &&  fabs(coordC[1*3+0]-coordC[2*3+0])<fabs(coordC[1*3+1]-coordC[2*3+1])){//special
                    wlist[0][0] = 0;
                    wlist[0][1] = 1;
                    wlist[1][0] = 1;
                    wlist[1][1] = 3;
                    wlist[2][0] = 4;
                    wlist[2][1] = 5;
                    wlist[3][0] = 5;
                    wlist[3][1] = 7;
                }
                //x-plane
                int w;
                int check[7];
                float warpingweight[7];
                float projectP[7][3];
                float pa[3], pb[3], pv[3], planeF[4], termL, termR, time;
                for(w=0; w<4; w++){
                    pa[0] = coordC[wlist[w][0]*3+0];
                    pa[1] = coordC[wlist[w][0]*3+1];
                    pa[2] = coordC[wlist[w][0]*3+2];
                    pb[0] = coordC[wlist[w][1]*3+0];
                    pb[1] = coordC[wlist[w][1]*3+1];
                    pb[2] = coordC[wlist[w][1]*3+2];
                    pv[0] = pa[0] - pb[0];
                    pv[1] = pa[1] - pb[1];
                    pv[2] = pa[2] - pb[2];
                    planeF[0] = 1;
                    planeF[1] = 0;
                    planeF[2] = 0;
                    planeF[3] = pt_T.x;
                    termL = pv[0]*planeF[0];
                    termR = planeF[3] - planeF[0]*pa[0];
                    time = termR/termL;
                    projectP[w][0] = pa[0]+pv[0]*time;
                    projectP[w][1] = pa[1]+pv[1]*time;
                    projectP[w][2] = pa[2]+pv[2]*time;
                    if( ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 1;
                    else if( ((pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))<(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 2;
                    else
                        check[w] = 0;
                    if(check[w]==0)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2),0.5);
                    else if(check[w]==1)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2),0.5);
                    else
                        warpingweight[w] = pow(pow(pa[0]-pb[0],2)+pow(pa[1]-pb[1],2)+pow(pa[2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5);
                    if(warpingweight[w]==0 || warpingweight[w]==1)
                        check[w] = 0;
                }
                //y-plane
                wlist[4][0] = 0;
                wlist[4][1] = 1;
                wlist[5][0] = 2;
                wlist[5][1] = 3;
                for(w=4; w<6; w++){
                    pa[0] = projectP[wlist[w][0]][0];
                    pa[1] = projectP[wlist[w][0]][1];
                    pa[2] = projectP[wlist[w][0]][2];
                    pb[0] = projectP[wlist[w][1]][0];
                    pb[1] = projectP[wlist[w][1]][1];
                    pb[2] = projectP[wlist[w][1]][2];
                    pv[0] = pa[0] - pb[0];
                    pv[1] = pa[1] - pb[1];
                    pv[2] = pa[2] - pb[2];
                    planeF[0] = 0;
                    planeF[1] = 1;
                    planeF[2] = 0;
                    planeF[3] = pt_T.y;
                    termL = pv[1]*planeF[1];
                    termR = planeF[3] - planeF[1]*pa[1];
                    time = termR/termL;
                    projectP[w][0] = pa[0]+pv[0]*time;
                    projectP[w][1] = pa[1]+pv[1]*time;
                    projectP[w][2] = pa[2]+pv[2]*time;
                    if( ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 1;
                    else if( ((pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))<(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 2;
                    else
                        check[w] = 0;
                    if(check[w]==0)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2),0.5);
                    else if(check[w]==1)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2),0.5);
                    else
                        warpingweight[w] = pow(pow(pa[0]-pb[0],2)+pow(pa[1]-pb[1],2)+pow(pa[2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5);
                    if(warpingweight[w]==0 || warpingweight[w]==1)
                        check[w] = 0;
                }
                //z-plane
                w = 6;
                pa[0] = projectP[4][0];
                pa[1] = projectP[4][1];
                pa[2] = projectP[4][2];
                pb[0] = projectP[5][0];
                pb[1] = projectP[5][1];
                pb[2] = projectP[5][2];
                pv[0] = pa[0] - pb[0];
                pv[1] = pa[1] - pb[1];
                pv[2] = pa[2] - pb[2];
                planeF[0] = 0;
                planeF[1] = 0;
                planeF[2] = 1;
                planeF[3] = pt_T.z;
                termL = pv[2]*planeF[2];
                termR = planeF[3] - planeF[2]*pa[2];
                time = termR/termL;
                projectP[w][0] = pa[0]+pv[0]*time;
                projectP[w][1] = pa[1]+pv[1]*time;
                projectP[w][2] = pa[2]+pv[2]*time;
                if( ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                    check[w] = 1;
                else if( ((pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))<(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                    check[w] = 2;
                else
                    check[w] = 0;
                if(check[w]==0)
                    warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2),0.5);
                else if(check[w]==1)
                    warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2),0.5);
                else
                    warpingweight[w] = pow(pow(pa[0]-pb[0],2)+pow(pa[1]-pb[1],2)+pow(pa[2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5);
                if(warpingweight[w]==0 || warpingweight[w]==1)
                    check[w] = 0;
                // get new point 
                for(w=0;w<4;w++){
                    pa[0] = coordNew[wlist[w][0]*3+0];
                    pa[1] = coordNew[wlist[w][0]*3+1];
                    pa[2] = coordNew[wlist[w][0]*3+2];
                    pb[0] = coordNew[wlist[w][1]*3+0];
                    pb[1] = coordNew[wlist[w][1]*3+1];
                    pb[2] = coordNew[wlist[w][1]*3+2];
                    if(check[w]==0){
                        projectP[w][0] = pa[0]*warpingweight[w]+pb[0]*(1-warpingweight[w]);
                        projectP[w][1] = pa[1]*warpingweight[w]+pb[1]*(1-warpingweight[w]);
                        projectP[w][2] = pa[2]*warpingweight[w]+pb[2]*(1-warpingweight[w]);
                    }
                    else if(check[w]==1){
                        projectP[w][0] = (pb[0]-pa[0]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][1] = (pb[1]-pa[1]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][2] = (pb[2]-pa[2]*warpingweight[w])/(1-warpingweight[w]);
                    }
                    else{
                        projectP[w][0] = (pa[0]-pb[0]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][1] = (pa[1]-pb[1]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][2] = (pa[2]-pb[2]*(1-warpingweight[w]))/warpingweight[w];
                    }
                }
                for(w=4;w<6;w++){
                    pa[0] = projectP[wlist[w][0]][0];
                    pa[1] = projectP[wlist[w][0]][1];
                    pa[2] = projectP[wlist[w][0]][2];
                    pb[0] = projectP[wlist[w][1]][0];
                    pb[1] = projectP[wlist[w][1]][1];
                    pb[2] = projectP[wlist[w][1]][2];
                    if(check[w]==0){
                        projectP[w][0] = pa[0]*warpingweight[w]+pb[0]*(1-warpingweight[w]);
                        projectP[w][1] = pa[1]*warpingweight[w]+pb[1]*(1-warpingweight[w]);
                        projectP[w][2] = pa[2]*warpingweight[w]+pb[2]*(1-warpingweight[w]);
                    }
                    else if(check[w]==1){
                        projectP[w][0] = (pb[0]-pa[0]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][1] = (pb[1]-pa[1]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][2] = (pb[2]-pa[2]*warpingweight[w])/(1-warpingweight[w]);
                    }
                    else{
                        projectP[w][0] = (pa[0]-pb[0]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][1] = (pa[1]-pb[1]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][2] = (pa[2]-pb[2]*(1-warpingweight[w]))/warpingweight[w];
                    }
                }
                w=6;
                pa[0] = projectP[4][0];
                pa[1] = projectP[4][1];
                pa[2] = projectP[4][2];
                pb[0] = projectP[5][0];
                pb[1] = projectP[5][1];
                pb[2] = projectP[5][2];
                if(check[w]==0){
                    projectP[w][0] = pa[0]*warpingweight[w]+pb[0]*(1-warpingweight[w]);
                    projectP[w][1] = pa[1]*warpingweight[w]+pb[1]*(1-warpingweight[w]);
                    projectP[w][2] = pa[2]*warpingweight[w]+pb[2]*(1-warpingweight[w]);
                }
                else if(check[w]==1){
                    projectP[w][0] = (pb[0]-pa[0]*warpingweight[w])/(1-warpingweight[w]);
                    projectP[w][1] = (pb[1]-pa[1]*warpingweight[w])/(1-warpingweight[w]);
                    projectP[w][2] = (pb[2]-pa[2]*warpingweight[w])/(1-warpingweight[w]);
                }
                else{
                    projectP[w][0] = (pa[0]-pb[0]*(1-warpingweight[w]))/warpingweight[w];
                    projectP[w][1] = (pa[1]-pb[1]*(1-warpingweight[w]))/warpingweight[w];
                    projectP[w][2] = (pa[2]-pb[2]*(1-warpingweight[w]))/warpingweight[w];
                }
                //get new position
                pt_T.x = projectP[w][0];
                pt_T.y = projectP[w][1];
                pt_T.z = projectP[w][2];
            }
            else{ // bp=10
                float dis1[2], dis3[2];
                int wlist[7][2];
                float line1[4] = {-coordC[4*3+1]+coordC[1*3+1], coordC[4*3+0]-coordC[1*3+0],0,0};
                float line2[4] = {-coordC[5*3+1]+coordC[0*3+1], coordC[5*3+0]-coordC[0*3+0],0,0};
                float line3[4] = {-coordC[6*3+1]+coordC[8*3+1], coordC[6*3+0]-coordC[8*3+0],0,0};
                line1[3] = -(line1[0]*coordC[4*3+0]+line1[1]*coordC[4*3+1]);
                line2[3] = -(line2[0]*coordC[0*3+0]+line2[1]*coordC[0*3+1]);
                line3[3] = -(line3[0]*coordC[6*3+0]+line3[1]*coordC[6*3+1]);
                if( (line1[0]*pt_T.x+line1[1]*pt_T.y+line1[2]*pt_T.z+line1[3])<0 ){
                    wlist[0][0] = 1;
                    wlist[1][0] = 2;
                    wlist[0][1] = 4;
                    wlist[1][1] = 3;
                    wlist[2][0] = 10;
                    wlist[3][0] = 11;
                    wlist[2][1] = 13;
                    wlist[3][1] = 12;
                }
                else if( (line2[0]*pt_T.x+line2[1]*pt_T.y+line2[2]*pt_T.z+line2[3])<0 ){
                    wlist[0][0] = 0;
                    wlist[1][0] = 1;
                    wlist[0][1] = 5;
                    wlist[1][1] = 4;
                    wlist[2][0] = 9;
                    wlist[3][0] = 10;
                    wlist[2][1] = 14;
                    wlist[3][1] = 13;
                }
                else if( (line3[0]*pt_T.x+line3[1]*pt_T.y+line3[2]*pt_T.z+line3[3])<0 ){
                    wlist[0][0] = 8;
                    wlist[1][0] = 0;
                    wlist[0][1] = 6;
                    wlist[1][1] = 5;
                    wlist[2][0] = 17;
                    wlist[3][0] = 9;
                    wlist[2][1] = 15;
                    wlist[3][1] = 14;
                }
                else{
                    wlist[0][0] = 7;
                    wlist[1][0] = 8;
                    wlist[0][1] = 6;
                    wlist[1][1] = 7;
                    wlist[2][0] = 16;
                    wlist[3][0] = 17;
                    wlist[2][1] = 15;
                    wlist[3][1] = 16;
                }
                //x-plane
                int w;
                int check[7];
                float warpingweight[7];
                float projectP[7][3];
                float pa[3], pb[3], pv[3], planeF[4], termL, termR, time;
                for(w=0; w<4; w++){
                    pa[0] = coordC[wlist[w][0]*3+0];
                    pa[1] = coordC[wlist[w][0]*3+1];
                    pa[2] = coordC[wlist[w][0]*3+2];
                    pb[0] = coordC[wlist[w][1]*3+0];
                    pb[1] = coordC[wlist[w][1]*3+1];
                    pb[2] = coordC[wlist[w][1]*3+2];
                    pv[0] = pa[0] - pb[0];
                    pv[1] = pa[1] - pb[1];
                    pv[2] = pa[2] - pb[2];
                    planeF[0] = 1;
                    planeF[1] = 0;
                    planeF[2] = 0;
                    planeF[3] = pt_T.x;
                    termL = pv[0]*planeF[0];
                    termR = planeF[3] - planeF[0]*pa[0];
                    time = termR/termL;
                    projectP[w][0] = pa[0]+pv[0]*time;
                    projectP[w][1] = pa[1]+pv[1]*time;
                    projectP[w][2] = pa[2]+pv[2]*time;
                    if( ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 1;
                    else if( ((pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))<(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 2;
                    else
                        check[w] = 0;
                    if(check[w]==0)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2),0.5);
                    else if(check[w]==1)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2),0.5);
                    else
                        warpingweight[w] = pow(pow(pa[0]-pb[0],2)+pow(pa[1]-pb[1],2)+pow(pa[2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5);
                    if(warpingweight[w]==0 || warpingweight[w]==1)
                        check[w] = 0;
                }
                //y-plane
                wlist[4][0] = 0;
                wlist[4][1] = 1;
                wlist[5][0] = 2;
                wlist[5][1] = 3;
                for(w=4; w<6; w++){
                    pa[0] = projectP[wlist[w][0]][0];
                    pa[1] = projectP[wlist[w][0]][1];
                    pa[2] = projectP[wlist[w][0]][2];
                    pb[0] = projectP[wlist[w][1]][0];
                    pb[1] = projectP[wlist[w][1]][1];
                    pb[2] = projectP[wlist[w][1]][2];
                    pv[0] = pa[0] - pb[0];
                    pv[1] = pa[1] - pb[1];
                    pv[2] = pa[2] - pb[2];
                    planeF[0] = 0;
                    planeF[1] = 1;
                    planeF[2] = 0;
                    planeF[3] = pt_T.y;
                    termL = pv[1]*planeF[1];
                    termR = planeF[3] - planeF[1]*pa[1];
                    time = termR/termL;
                    projectP[w][0] = pa[0]+pv[0]*time;
                    projectP[w][1] = pa[1]+pv[1]*time;
                    projectP[w][2] = pa[2]+pv[2]*time;
                    if( ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 1;
                    else if( ((pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                    && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))<(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                        check[w] = 2;
                    else
                        check[w] = 0;
                    if(check[w]==0)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2),0.5);
                    else if(check[w]==1)
                        warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2),0.5);
                    else
                        warpingweight[w] = pow(pow(pa[0]-pb[0],2)+pow(pa[1]-pb[1],2)+pow(pa[2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5);
                    if(warpingweight[w]==0 || warpingweight[w]==1)
                        check[w] = 0;
                }
                //z-plane
                w = 6;
                pa[0] = projectP[4][0];
                pa[1] = projectP[4][1];
                pa[2] = projectP[4][2];
                pb[0] = projectP[5][0];
                pb[1] = projectP[5][1];
                pb[2] = projectP[5][2];
                pv[0] = pa[0] - pb[0];
                pv[1] = pa[1] - pb[1];
                pv[2] = pa[2] - pb[2];
                planeF[0] = 0;
                planeF[1] = 0;
                planeF[2] = 1;
                planeF[3] = pt_T.z;
                termL = pv[2]*planeF[2];
                termR = planeF[3] - planeF[2]*pa[2];
                time = termR/termL;
                projectP[w][0] = pa[0]+pv[0]*time;
                projectP[w][1] = pa[1]+pv[1]*time;
                projectP[w][2] = pa[2]+pv[2]*time;
                if( ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))>(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                    check[w] = 1;
                else if( ((pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))>(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2))) 
                && ((pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2))<(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2))) )
                    check[w] = 2;
                else
                    check[w] = 0;
                if(check[w]==0)
                    warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(pb[0]-pa[0],2)+pow(pb[1]-pa[1],2)+pow(pb[2]-pa[2],2),0.5);
                else if(check[w]==1)
                    warpingweight[w] = pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pa[0],2)+pow(projectP[w][1]-pa[1],2)+pow(projectP[w][2]-pa[2],2),0.5);
                else
                    warpingweight[w] = pow(pow(pa[0]-pb[0],2)+pow(pa[1]-pb[1],2)+pow(pa[2]-pb[2],2),0.5) / pow(pow(projectP[w][0]-pb[0],2)+pow(projectP[w][1]-pb[1],2)+pow(projectP[w][2]-pb[2],2),0.5);
                if(warpingweight[w]==0 || warpingweight[w]==1)
                    check[w] = 0;
                // get new point 
                for(w=0;w<4;w++){
                    pa[0] = coordNew[wlist[w][0]*3+0];
                    pa[1] = coordNew[wlist[w][0]*3+1];
                    pa[2] = coordNew[wlist[w][0]*3+2];
                    pb[0] = coordNew[wlist[w][1]*3+0];
                    pb[1] = coordNew[wlist[w][1]*3+1];
                    pb[2] = coordNew[wlist[w][1]*3+2];
                    if(check[w]==0){
                        projectP[w][0] = pa[0]*warpingweight[w]+pb[0]*(1-warpingweight[w]);
                        projectP[w][1] = pa[1]*warpingweight[w]+pb[1]*(1-warpingweight[w]);
                        projectP[w][2] = pa[2]*warpingweight[w]+pb[2]*(1-warpingweight[w]);
                    }
                    else if(check[w]==1){
                        projectP[w][0] = (pb[0]-pa[0]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][1] = (pb[1]-pa[1]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][2] = (pb[2]-pa[2]*warpingweight[w])/(1-warpingweight[w]);
                    }
                    else{
                        projectP[w][0] = (pa[0]-pb[0]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][1] = (pa[1]-pb[1]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][2] = (pa[2]-pb[2]*(1-warpingweight[w]))/warpingweight[w];
                    }
                }
                for(w=4;w<6;w++){
                    pa[0] = projectP[wlist[w][0]][0];
                    pa[1] = projectP[wlist[w][0]][1];
                    pa[2] = projectP[wlist[w][0]][2];
                    pb[0] = projectP[wlist[w][1]][0];
                    pb[1] = projectP[wlist[w][1]][1];
                    pb[2] = projectP[wlist[w][1]][2];
                    if(check[w]==0){
                        projectP[w][0] = pa[0]*warpingweight[w]+pb[0]*(1-warpingweight[w]);
                        projectP[w][1] = pa[1]*warpingweight[w]+pb[1]*(1-warpingweight[w]);
                        projectP[w][2] = pa[2]*warpingweight[w]+pb[2]*(1-warpingweight[w]);
                    }
                    else if(check[w]==1){
                        projectP[w][0] = (pb[0]-pa[0]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][1] = (pb[1]-pa[1]*warpingweight[w])/(1-warpingweight[w]);
                        projectP[w][2] = (pb[2]-pa[2]*warpingweight[w])/(1-warpingweight[w]);
                    }
                    else{
                        projectP[w][0] = (pa[0]-pb[0]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][1] = (pa[1]-pb[1]*(1-warpingweight[w]))/warpingweight[w];
                        projectP[w][2] = (pa[2]-pb[2]*(1-warpingweight[w]))/warpingweight[w];
                    }
                }
                w=6;
                pa[0] = projectP[4][0];
                pa[1] = projectP[4][1];
                pa[2] = projectP[4][2];
                pb[0] = projectP[5][0];
                pb[1] = projectP[5][1];
                pb[2] = projectP[5][2];
                if(check[w]==0){
                    projectP[w][0] = pa[0]*warpingweight[w]+pb[0]*(1-warpingweight[w]);
                    projectP[w][1] = pa[1]*warpingweight[w]+pb[1]*(1-warpingweight[w]);
                    projectP[w][2] = pa[2]*warpingweight[w]+pb[2]*(1-warpingweight[w]);
                }
                else if(check[w]==1){
                    projectP[w][0] = (pb[0]-pa[0]*warpingweight[w])/(1-warpingweight[w]);
                    projectP[w][1] = (pb[1]-pa[1]*warpingweight[w])/(1-warpingweight[w]);
                    projectP[w][2] = (pb[2]-pa[2]*warpingweight[w])/(1-warpingweight[w]);
                }
                else{
                    projectP[w][0] = (pa[0]-pb[0]*(1-warpingweight[w]))/warpingweight[w];
                    projectP[w][1] = (pa[1]-pb[1]*(1-warpingweight[w]))/warpingweight[w];
                    projectP[w][2] = (pa[2]-pb[2]*(1-warpingweight[w]))/warpingweight[w];
                }
                //get new position
                pt_T.x = projectP[w][0];
                pt_T.y = projectP[w][1];
                pt_T.z = projectP[w][2];
            }

            // Project onto Image
            pix.x = convert_int(round((pt_T.x/fabs(pt_T.z))*calib[0] + calib[2])); 
            pix.y = convert_int(round((pt_T.y/fabs(pt_T.z))*calib[4] + calib[5])); 
            
            // Check if the pixel is in the frame
            if (pix.x < 0 || pix.x > m_col-1 || pix.y < 0 || pix.y > n_row-1){
                if (Weight[idx] == 0)
                    TSDF[idx] = (short int)(convVal);
                continue;
            }
            
            //Compute distance between project voxel and surface in the RGBD image
            float dist = -(pt_T.z - Depth[pix.x + m_col*pix.y])/nu;
            dist = min(1.0f, max(-1.0f, dist));            
            if (Depth[pix.x + m_col*pix.y] == 0) {
                if (Weight[idx] == 0)
                    TSDF[idx] = (short int)(convVal);
                continue;
            }
            
            if (dist > 1.0f) dist = 1.0f;
            else dist = max(-1.0f, dist);
                
            // Running Average
            float prev_tsdf = (float)(TSDF[idx])/convVal;
            float prev_weight = (float)(Weight[idx]);
            
            TSDF[idx] =  (short int)(round(((prev_tsdf*prev_weight+dist)/(prev_weight+1.0f))*convVal));
            Weight[idx] = min(1000, Weight[idx]+1);
         }
        
}
"""



