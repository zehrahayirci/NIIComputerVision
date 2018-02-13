"""
@Author Inoe ANDRE
Cross module functions
"""

import numpy as np
from numpy import linalg as LA




def in_mat_zero2one(mat):
    """
    Replace in the matrix all the 0 to 1
    :param mat: input matrix containing 0
    :return:  mat with 1 instead of 0
    """
    mat_tmp = (mat != 0.0)
    res = mat * mat_tmp + ~mat_tmp
    return res



def InvPose(Pose):
    """
    Compute the inverse transform of Pose
    :param Pose: 4*4 Matrix of the camera pose
    :return: matrix containing the inverse transform of Pose
    y = R*x + T
    x = R^(-1)*y - R^(-1)*T
    """
    PoseInv = np.zeros(Pose.shape, Pose.dtype)
    # Inverse rotation part R^(-1)
    PoseInv[0:3, 0:3] = LA.inv(Pose[0:3, 0:3])
    # Inverse Translation part R^(-1)*T
    PoseInv[0:3, 3] = -np.dot(PoseInv[0:3, 0:3], Pose[0:3, 3])
    PoseInv[3, 3] = 1.0
    return PoseInv

def normalized_cross_prod(a, b):
    '''
    Compute the cross product of 2 vectors and normalized it
    :param a: first 3 elements vector
    :param b: second 3 elements vector
    :return: the normalized cross product between 2 vector
    '''
    res = np.zeros(3, dtype="float")
    if (LA.norm(a) == 0.0 or LA.norm(b) == 0.0):
        return res
    # normalized a and b
    a = a / LA.norm(a)
    b = b / LA.norm(b)
    # compute cross product
    res[0] = a[1] * b[2] - a[2] * b[1]
    res[1] = -a[0] * b[2] + a[2] * b[0]
    res[2] = a[0] * b[1] - a[1] * b[0]
    # normalized result
    if (LA.norm(res) > 0.0):
        res = res / LA.norm(res)
    return res


def division_by_norm(mat, norm):
    '''
    This fonction divide a n by m by p=3 matrix, point by point, by the norm made through the p dimension
    It ignores division that makes infinite values or overflow to replace it by the former mat values or by 0
    :param mat:
    :param norm:
    :return:
    '''
    for i in range(3):
        with np.errstate(divide='ignore', invalid='ignore'):
            mat[:, :, i] = np.true_divide(mat[:, :, i], norm)
            mat[:, :, i][mat[:, :, i] == np.inf] = 0
            mat[:, :, i] = np.nan_to_num(mat[:, :, i])
    return mat


def normalized_cross_prod_optimize(a, b):
    """
    Compute the cross product of list of 2 vectors and normalized it
    :param a: first 3 elements vector
    :param b: second 3 elements vector
    :return: the normalized cross product between 2 vector
    """
    # res = np.zeros(a.Size, dtype = "float")
    norm_mat_a = np.sqrt(np.sum(a * a, axis=2))
    norm_mat_b = np.sqrt(np.sum(b * b, axis=2))
    # changing every 0 to 1 in the matrix so that the division does not generate nan or infinite values
    norm_mat_a = in_mat_zero2one(norm_mat_a)
    norm_mat_b = in_mat_zero2one(norm_mat_b)
    # compute a/ norm_mat_a
    a = division_by_norm(a, norm_mat_a)
    b = division_by_norm(b, norm_mat_b)
    # compute cross product with matrix
    res = np.cross(a, b)
    # compute the norm of res using the same method for a and b
    norm_mat_res = np.sqrt(np.sum(res * res, axis=2))
    norm_mat_res = in_mat_zero2one(norm_mat_res)
    # norm division
    res = division_by_norm(res, norm_mat_res)
    return res

def getConnectBP(bp):
    '''
    return the list of connected body part index
    :param bp: the index of body part
    :retrun: connected bp list
    '''  
    if bp==1:
        bp_n = [2,12]
    elif bp==2:
        bp_n = [1, 9]
    elif bp==3:
        bp_n = [4, 11]
    elif bp==4:
        bp_n = [3, 9]
    elif bp==5:
        bp_n = [6, 7,10]
    elif bp==6:
        bp_n = [5, 13]
    elif bp==7:
        bp_n = [5, 8, 10]
    elif bp==8:
        bp_n = [7, 14]
    elif bp==9:
        bp_n = [2,4,10]
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
    
    return bp_n

def getBodypartPoseIndex(bp):
    '''
    return the list of junction index of body part
    :param bp: the index of body part
    :retrun: connected bp list
    ''' 
    if bp==1:
        pos = [5,6]
    if bp==2:
        pos = [4,5]
    if bp==3:
        pos = [9, 10]
    if bp==4:
        pos = [8,9]
    if bp==5:
        pos = [16,17]
    if bp==6:
        pos = [17, 18]
    if bp==7:
        pos = [12,13]
    if bp==8:
        pos = [13,14]
    if bp==9:
        pos = [3,2]
    if bp==10:
        pos = [20, 1, 0, 4, 8]
    if bp==11:
        pos = [11]
    if bp==12:
        pos = [7]
    if bp==13:
        pos = [15]
    if bp==14:
        pos = [19]
    return pos

def GetRotatefrom2Vector(a, b):
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
    if c == -1:
        print "meet a=-b"
        return np.identity(3)
    R += Av + 1/(1+c)*np.dot(Av,Av)
    return R

'''Dual Quaternion'''
def getQuaternionfromMatrix(mat):
    quat = np.zeros(4, dtype=np.float32)
    tr = mat[0,0]+mat[1,1]+mat[2,2]
    if tr>0:
        S = pow(tr+1, 0.5)*2
        quat[0] = 0.25*S
        quat[1] = (mat[2,1]-mat[1,2])/S
        quat[2] = (mat[0,2]-mat[2,0])/S
        quat[3] = (mat[1,0]-mat[0,1])/S
    elif (mat[0,0]>mat[1,1]) and (mat[0,0]>mat[2,2]):
        S = pow(1+mat[0,0]-mat[1,1]-mat[2,2], 0.5)*2
        quat[0] = (mat[2,1]-mat[1,2])/S
        quat[1] = 0.25*S
        quat[2] = (mat[0,1]+mat[1,0])/S
        quat[3] = (mat[0,2]+mat[2,0])/S
    elif mat[1,1]>mat[2,2]:
        S = pow(1+mat[1,1]-mat[0,0]-mat[2,2],0.5)*2
        quat[0] = (mat[0,2]-mat[2,0])/S
        quat[1] = (mat[0,1]+mat[1,0])/S
        quat[2] = 0.25*S
        quat[3] = (mat[1,2]+mat[2,1])/S
    else:
        S = pow(1+mat[2,2]-mat[0,0]-mat[1,1], 0.5)*2
        quat[0] = (mat[1,0]-mat[0,1])/S
        quat[1] = (mat[0,2]+mat[2,0])/S
        quat[2] = (mat[1,2]+mat[2,1])/S
        quat[3] = 0.25*S
    return quat

def getQuaternionConj(quat):
    quat[1:4] *= -1
    return quat

def getQuaternionMul(quat1, quat2):
    quat = np.zeros(4, dtype = np.float32)
    quat[0] = quat1[0]*quat2[0] - np.dot(quat1[1:4], quat2[1:4])
    quat[1:4] = quat1[0]*quat2[1:4]+quat2[0]*quat1[1:4]+np.cross(quat1[1:4], quat2[1:4])
    return quat

def getQuaternionNormalize(quat):
    if np.dot(quat,quat)!= 0:
        return quat/pow(np.dot(quat,quat),1/2)
    return 0

def getDualQuaternionfromMatrix(mat):
    quatReal = getQuaternionfromMatrix(mat)
    quatReal = getQuaternionNormalize(quatReal)
    quatDual = np.zeros(4, dtype=np.float32)
    quatDual[1:4] = mat[0:3,3]
    quatDual = 0.5*getQuaternionMul(quatDual, quatReal)
    dquat = np.zeros((2,4), dtype=np.float32)
    dquat[0,:] = quatReal
    dquat[1,:] = quatDual
    return dquat

def getDualQuaternionConj(dquat):
    dquat[0,:] = getQuaternionConj(dquat[0,:])
    dquat[1,:] = getQuaternionConj(dquat[1,:])
    return dquat

def getDualQuaternionMul(dquat1, dquat2):
    dquat = np.zeros((2,4), dtype=np.float32)
    dquat[0,:] = getQuaternionMul(dquat1[0,:], dquat2[0,:])
    dquat[1,:] = getQuaternionMul(dquat1[0,:], dquat2[1,:])
    dquat[1,:] += getQuaternionMul(dquat1[1,:], dquat2[0,:])
    return dquat

def getDualQuaternionNormalize(dquat):
    mag = dquat[0][0]*dquat[0][0]+dquat[0][1]*dquat[0][1]
    mag += dquat[0][2]*dquat[0][2]+dquat[0][3]*dquat[0][3]
    mag = pow(float(mag), 0.5)
    if mag!=0:
        return dquat/mag
    return 0

def getMatrixfromDualQuaternion(dquat):
    dquat = dquat.reshape((2,4))
    dquat = getDualQuaternionNormalize(dquat)
    mat = np.identity(4, dtype=np.float32)

    mat[0,0] = dquat[0,0]*dquat[0,0]+dquat[0,1]*dquat[0,1]-dquat[0,2]*dquat[0,2]-dquat[0,3]*dquat[0,3]
    mat[0,1] = 2*dquat[0,1]*dquat[0,2] + 2*dquat[0,0]*dquat[0,3]
    mat[0,2] = 2*dquat[0,1]*dquat[0,3] - 2*dquat[0,0]*dquat[0,2]

    mat[1,0] = 2*dquat[0,1]*dquat[0,2] - 2*dquat[0,0]*dquat[0,3]
    mat[1,1] = dquat[0,0]*dquat[0,0] + dquat[0,2]*dquat[0,2] - dquat[0,1]*dquat[0,1] - dquat[0,3]*dquat[0,3]
    mat[1,2] = 2*dquat[0,2]*dquat[0,3] + 2*dquat[0,0]*dquat[0,1]

    mat[2,0] = 2*dquat[0,1]*dquat[0,3] + 2*dquat[0,0]*dquat[0,2]
    mat[2,1] = 2*dquat[0,2]*dquat[0,3] - 2*dquat[0,0]*dquat[0,1]
    mat[2,2] = dquat[0,0]*dquat[0,0] + dquat[0,3]*dquat[0,3] - dquat[0,1]*dquat[0,1] - dquat[0,2]*dquat[0,2]

    quat = getQuaternionMul(dquat[1,:]*2,getQuaternionConj(dquat[0,:]))
    mat[3,0] = quat[1]
    mat[3,1] = quat[2]
    mat[3,2] = quat[3]

    return mat.T