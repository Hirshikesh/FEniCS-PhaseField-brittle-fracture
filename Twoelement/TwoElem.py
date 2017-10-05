# --------------------------------
# purpose: phase field for fracture
# Benchmark problem
#  Hirshikesh, Sundarajan Natarajan, Ratna Kumar Annabatutla 
#  IITM, Aug 2017
#-------------------------------------
from dolfin import *
import numpy as np
import math
from getMatMtx import D_Matrix
parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True
import matplotlib.pyplot as plt
set_log_active(False)
#------------------------------------------------
#          Define Geometry
#------------------------------------------------

# Define mesh
mesh = UnitSquareMesh(1,1)
plot(mesh)
interactive()
stressState = 'PlaneStrain'
ndim = mesh.geometry().dim()
#------------------------------------------------
#          Define Space
#------------------------------------------------

V = FunctionSpace(mesh,'CG',1)
W = VectorFunctionSpace(mesh,'CG',1)

p , q = TrialFunction(V), TestFunction(V)
u , v , du =  TrialFunction(W), TestFunction(W), Function(W)
#------------------------------------------------
#            material Parameters
#------------------------------------------------
l_fac = 2
Gc =  5
hm = mesh.hmin()
l_o = l_fac*hm
l_o = 0.1

E = 210e3
nu = 0.3

lmbda, mu = (E*nu/((1.0 + nu )*(1.0-2.0*nu))) , (E/(2*(1+nu)))

#------------------------------------------------
#           Classes
#------------------------------------------------

class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]-1) < tol and on_boundary

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and on_boundary


def epsilon(v):
    return 0.5*(grad(v)  + grad(v).T)
def eps(u):
    return sym(grad(u))

def sigma(u):
    return 2.0*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(2)

def en_dens(u):
    str_ele = 0.5*(grad(u) + grad(u).T)
    IC = tr(str_ele)
    ICC = tr(str_ele * str_ele)
    return (0.5*lmbda*IC**2) + mu*ICC
kn = lmbda + mu
def hist(u):
    return 0.5*kn*( 0.5*(tr(eps(u)) + abs(tr(eps(u)))) )**2 + mu*tr(dev(eps(u))*dev(eps(u)))
    

#--------------------------------------------------------
#    Boundary Conditions
#-------------------------------------------------------

Top = top()
Bottom = bottom()

u_Lx = Expression("t",t = 0.0)
ftx =  DirichletBC(W.sub(1),u_Lx,Top)
fty =  DirichletBC(W.sub(0),Constant(0.0),Top)

fbx =  DirichletBC(W.sub(0),Constant(0.0),Bottom)
fby =  DirichletBC(W.sub(1),Constant(0.0),Bottom)

bc_disp = [ftx, fty, fbx, fby ]

# initialization 

u_old,u_conv, unew = Function(W),Function(W), Function(W)
phi_old,phi_conv = Function(V),Function(V)

#-------------------------------------------------------
#  Get material matrix plane stress or plane strain
#  for stress caluclation
#-------------------------------------------------------
D_mat = D_Matrix(E,nu,stressState)  
D_mat = as_matrix(D_mat)
#-------------------------------------------------------
# get strian in vector form
# for stress calculation
#------------------------------------------------------
def epsC(u):
        return as_vector([u[i].dx(i) for i in range(2)] +[u[i].dx(j) + u[j].dx(i) for (i,j) in [(0,1)]])

def eps(v):
    return sym(grad(v))

def eps_dev(u_old):
   return dev(eps(u_old))

#--------------------------------------------------------
#               Define Variational Problem
#--------------------------------------------------------

E_du = ((1 - phi_old)**2 + 1e-6)*inner(grad(v),sigma(u))*dxwhile 

E_phi = ( Gc*l_o*inner(grad(p),grad(q))+\
            ((Gc/l_o) + 2.*hist(unew))*inner(p,q)-\
            2.*hist(unew)*q)*dx
du = Function(W)
p = Function(V)

t = 1e-3
max_load = 0.1
deltaT = 1e-3
ut = 1

#------------------------------------- 
# displacement controlled test 
# starts here
#------------------------------------
with open('TwoElem.txt', 'a') as f:
    while t <= max_load:
        u_Lx.t=t*ut
        iter = 1
        toll = 1e-2
        err = 1
        maxiter = 100
        while err > toll:
        
            solve(lhs(E_du)==rhs(E_du), du, bc_disp)
            unew.assign(du)
            solve(lhs(E_phi)==rhs(E_phi),p)
            err_u = errornorm(du,u_old, norm_type = 'l2', mesh = None)/norm(du)
            err_phi = errornorm(p,phi_old, norm_type = 'l2', mesh = None)/norm(p)
            
            err = max(err_u,err_phi)
            u_old.assign(du)
            phi_old.assign(p)
            iter = iter + 1
            if err < toll:
            #plot(du, key = 'disp', title = 'disp')
            #print 'Solution converge after',iter
                plot(p, key = 'p', title = 'phi%.4f'%(t))
                A = Point(1,1)
                y1 = phi_old(A)
                #plt.plot(t,y1, 'sr', label = 'phi')
                dd = (t**2*(lmbda*((1-0.3)/0.3)))/(Gc/l_o + t**2*(lmbda*((1-0.3)/0.3)))
                
                Str = (lmbda*((1-0.3)/0.3))*t*(1-dd)**2
                sigma_w = D_mat*epsC(u_old)
                sigmaT = project(sigma_w)
                [s0,s1,s2] = sigmaT.split(True)
                y2 = s1(A)
                y3 = y2*(1-y1)**2

                a = np.array([Str,y3])
                #np.savetxt(f,a)
                plt.figure(1)
                plt.subplot(211)
                plt.plot(t, Str, 'bo')
                plt.plot(t, y3, 'k*')
                plt.subplot(212)
                AAA  = ((Str-y3))
                plt.plot(t, y1, 'ro')
                
                plt.pause(0.05)
            
        t+=deltaT

plt.show()

print "Simulation done with no error :)"


