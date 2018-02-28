from fab_model import c_probs
#from sonic_data import interp_pvel 
from params import *
import numpy as np
import scipy as sp


''' elastic constants '''
A=14.06*10**9
K=15.24*10**9
L=3.06*10**9
N=3.455*10**9
F=5.88*10**9
rho = 917
c11 = A#14.06*10**9#13.2*10**9 # xx xx
c12 = A-2*N#6.69*10**9 # xx yy
c13 = F#5.84*10**9 # xx zz
c33 = K#15.24*10**9#14.42*10**9
c22 = c11
c23 = c13
c44 = L#2.89*10**9
c66 = N#(c11-c12)/2
char_time =(3.0*1e3)**-1
#divide nondim vels by char_time
#to get back regular velocities
#depths_pvels = np.genfromtxt("../data/NEEM_3m_avr.txt",skip_header=True).astype("float32")
#depths_pvels = depths_pvels[depths_pvels[:,0]>250,:]
#depths = depths_pvels[:,0]
#nvels = depths.size
#ntrain = int(np.ceil(depths.max() - depths.min()))
#depths_train = np.linspace(depths.min(),depths.max(),ntrain)[:,np.newaxis]
#interp_pvel =  interp1d(depths_at_vels[:,0],pvel[:,0])
#pvels_train = interp_pvel(depths_train)

#vels_dimensional = np.hstack([depths_pvels[:,1][:,np.newaxis],np.zeros((nvels,2))])
#vels_train_nd = np.hstack([pvels_train,np.zeros((ntrain,2))]) * char_time
#vels_nd = vels_dimensional * char_time
#sqvels_train = vels_train_nd**2
#sqvels_wh = (sqvels-sqvels.mean())/(wh.Xstd[0]+1e-15)+sqvels.mean()
chris_meps = np.load("../data/chris_meps.npy").astype('float32')
#chris_meps_wh = (chris_meps/wh.Xstd[0]).astype('float32')


def rotC(C,theta,phi):
    ''' returns a rotated stiffness matrix '''
    R = rot(theta, phi)
    Cp = np.zeros((6,6)) 
    for a in range(0,6):
        for b in range(0,6):
            #print a
            i,j = indexBackConvert(a)
            k,l = indexBackConvert(b)

            for p in range(0,3):
                for q in range(0,3):
                    for r in range(0,3):
                        for s in range(0,3):
                           c = indexConvert(p,q)
                           d = indexConvert(r,s)
                           Cp[a,b] = Cp[a,b] + R[i,p]*R[j,q]*R[k,r]*R[l,s]*C[c,d]

    return Cp



def indexConvert(i,j): #for C rotation
    i = i+1
    j = j+1
    if i == j:
        return i-1
    else:
        return 8-(i+j)


def indexBackConvert(a): #for C rotation
    a = a+1
    if a == 1:
        i=1
        j=1
    elif a==2:
        i=2
        j=2
    elif a==3:
        i=3
        j=3
    elif a==4:
        i=2
        j=3
    elif a==5:
        i=1
        j=3
    elif a==6:
        i=1
        j=2
    i = i-1
    j = j-1
    return i,j


def rot(theta, phi):
    ''' returns a rotation matrix  '''
    # rotate 'phi' around y-axis, the 'theta' around z-axis
#    Rx = array([[1,0,0],
#          [0,cos(theta),-sin(theta)],
#          [0,sin(theta),cos(theta)]])

    Rz = np.array([[np.cos(theta),-np.sin(theta),0],
          [np.sin(theta),np.cos(theta),0],
           [0,0,1]])

    Ry = np.array([[np.cos(phi), 0, np.sin(phi)],
           [0,1,0],
            [-np.sin(phi),0,np.cos(phi)]])
    R = np.inner(Rz, Ry)
    return R




''' THis will represent elasticity for a hexagonally symmetric crystal with th-axis along the
z direction '''
Cvert = np.array([[c11, c12, c13, 0,0,0],
[c12, c22,c23,0,0,0],
[c13,c23,c33,0,0,0],
[0,0,0,c44,0,0],
[0,0,0,0,c44,0],
[0,0,0,0,0,c66]])*char_time**2/rho

def meps_to_chrisMats(meps):
    n = meps.shape[0]
    phi = np.arccos(meps[:,2])
    theta = np.arctan2(meps[:,1],meps[:,0])
    chrisMats = np.zeros((n,3,3))
    for i in xrange(0,n):
        C = rotC(Cvert,theta[i],phi[i])
	c55 = C[4,4]
	c45 = C[3,4]
	c35 = C[2,4]
	c44 = C[3,3]
	c34 = C[2,3]
	c33 = C[2,2]
        chrisMats[i,:,:] = [[c55,c45,c35],[c45,c44,c34],[c35,c34,c33]]
    return chrisMats
        


