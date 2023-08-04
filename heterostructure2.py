import numpy as np
from numpy.linalg import det
from numba import njit
pi=np.pi
alpha=1./137
c=299792458


class Crystal:
    def __init__(self,number_of_layers,dielectrics,first_dielectric,widths,sig):
        self.number_of_layers=number_of_layers
        self.dielectric_composition=dielectrics
        self.dielectric_width=widths
        self.first_dielectric=first_dielectric
        self.conductivities=sig
        
            
def interface_matrix(eps1,eps2,cosn,cosa,sig=0,mode='TM'):
    mat=np.zeros([2,2],dtype=np.complex128)
    
    if(mode=='TM'):
        r=np.sqrt(eps2/eps1)*cosa/cosn
        fac=pi*alpha*sig/np.sqrt(eps2)*cosn
        mat[0,0]=1+r+fac
        mat[0,1]=1-r-fac #sign change
        mat[1,0]=1-r+fac #sign change
        mat[1,1]=1+r-fac
    elif(mode=='TE'):
        r=np.sqrt(eps1/eps2)*cosa/cosn
        fac=pi*alpha*sig/np.sqrt(eps2)/cosn
        mat[0,0]=1+r-fac
        mat[0,1]=1-r-fac #sign change
        mat[1,0]=1-r+fac #sign change
        mat[1,1]=1+r+fac        
    return mat/2


def prop_die(k,w):
    mat=np.zeros([2,2],dtype=np.complex128)
    phase=k*w
    mat[0,0]=np.exp(1.j*phase)
    mat[1,1]=np.exp(-1.j*phase)
    return mat


#Write TE code

def prop(Crystal,freq,q,mode='TM'):   
    M=np.identity(2,dtype=np.complex128)
    eps_a=Crystal.first_dielectric(freq)
    for j in range(Crystal.number_of_layers):
        eps=Crystal.dielectric_composition[j](freq)
        w=Crystal.dielectric_width[j]
        sig=Crystal.conductivities[j](freq)
        k0=freq/c
        p=np.sqrt(eps)*k0
        k=np.sqrt(p**2-q**2+0.j)
        p_a=np.sqrt(eps_a)*k0
        k_a=np.sqrt(p_a**2-q**2+0.j)
        cosa=k/p
        cosn=k_a/p_a        
#        theta=np.arcsin( np.sqrt( eps_a/eps)*np.sin(theta_a) +0.j)
        Mj=prop_die(k,w)
        Mj2=interface_matrix(eps_a,eps,cosn,cosa,sig,mode)
        M=Mj.dot(Mj2.dot(M))                     
        eps_a=eps
    return M

def fresnel(transfer_matrix):
    detM=det(transfer_matrix)
    a=transfer_matrix[0,0]
    d=transfer_matrix[1,1]
    c=transfer_matrix[1,0]
    b=transfer_matrix[0,1]
    r1=-c/d
    t1=detM/d
    t2=1/a#1/d
    r2=c/a#b/d
    return r1,t1,r2,t2
    
def loss_function(qvec,fvec,Cr,mode='TM'):
    Nq=len(qvec)
    Nf=len(fvec)
    Loss=np.zeros([Nf,Nq])
    
    for i in range(Nq):
        for j in range(Nf):
            freq=fvec[j]
            q=qvec[i]
            Mtemp=prop(Cr,freq,q,mode)
            r1,t1,r2,t2=fresnel(Mtemp)
            Loss[Nf-j-1,i]=r2.imag
    return Loss