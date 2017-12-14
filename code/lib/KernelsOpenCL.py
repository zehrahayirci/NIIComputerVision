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
                           __constant float *BBTrans, const int BBNum, __constant float *coords, __global float *tempPose,  
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
            int BBc, Trc;
            float weight, tempweight;
            float Tr[16]={0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1};
            float sumweight = 0;
            for( BBc=0; BBc<BBNum; BBc++){
                weight = pow(pt.x-coords[0+BBc*3],2);
                weight += pow(pt.y-coords[1+BBc*3],2);
                weight += pow(pt.z-coords[2+BBc*3],2);
                tempweight = weight;
                weight = 1/pow(weight,0.5);
                sumweight += weight;
                for (Trc=0; Trc<12; Trc++){
                    Tr[Trc] += weight * BBTrans[Trc+16*BBc];
                }
            }
            for (Trc=0; Trc<12; Trc++){
                Tr[Trc] /= sumweight;
            }
            pt=pt_T;
            pt_T.x =  Tr[0]*pt.x + Tr[1]*pt.y + Tr[2]*pt.z + Tr[3];
            pt_T.y =  Tr[4]*pt.x + Tr[5]*pt.y + Tr[6]*pt.z + Tr[7];
            pt_T.z =  Tr[8]*pt.x + Tr[9]*pt.y + Tr[10]*pt.z + Tr[11];

            if(x==0 && y==0 && z==0){
                int temp;
                for(temp=0; temp<16; temp++){
                    tempPose[temp] = Tr[temp];
                }
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



