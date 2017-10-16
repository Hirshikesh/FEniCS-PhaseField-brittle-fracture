# --------------------------------
# purpose: phase field for fracture
# Tension test 3D
#  Hirshikesh, Sundarajan Natarajan, Ratna Kumar Annabatutla 
#  IITM, Aug 2017
#-------------------------------------
from dolfin import *
import numpy as np
import math

parameters['form_compiler']['cpp_optimize'] = True
parameters['form_compiler']['optimize'] = True

set_log_active(False)
#------------------------------------------------
#          Define Geometry
#------------------------------------------------

mesh = Mesh('threeDEdgeCrackPart.xml')

print 'lo', 2*mesh.hmin()
print 'cells', mesh.num_cells()

plot(mesh)
interactive()
ndim = mesh.geometry().dim()

#------------------------------------------------
#          Define Space
#------------------------------------------------

V = FunctionSpace(mesh,'CG',1)
W = VectorFunctionSpace(mesh,'CG',1)

p , q = TrialFunction(V), TestFunction(V)
u , v = TrialFunction(W), TestFunction(W)
#------------------------------------------------
#           Parameters
#------------------------------------------------
l_fac = 2
Gc =  0.5
hm = mesh.hmin()
l_o = l_fac*hm


E = 20.8e3
nu = 0.3

lmbda, mu = Constant(E*nu/((1.0 + nu )*(1.0-2.0*nu))) , Constant(E/(2*(1+nu)))

#------------------------------------------------
#           Classes
#------------------------------------------------

class top(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]-10) < tol and on_boundary

class bottom(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and on_boundary
class bottomL(SubDomain):
    def inside(self,x,on_boundary):
        tol = 1e-10
        return abs(x[1]) < tol and x[0]==0.0 and x[2]==0.0


def epsilon(v):
    return sym(grad(v))

def sigma(u):
    return 2.0*mu*epsilon(u) + lmbda*tr(epsilon(u))*Identity(3)

def en_dens(u):
    str_ele = 0.5*(grad(u) + grad(u).T)
    IC = tr(str_ele)
    ICC = tr(str_ele * str_ele)
    return (0.5*lmbda*IC**2) + mu*ICC

#--------------------------------------------------------
#    Boundary Conditions
#-------------------------------------------------------

Top = top()
Bottom = bottom()
BottomL = bottomL()
u_Lx = Expression("t",t = 0.00001)
fty =  DirichletBC(W.sub(1),u_Lx,Top)

fbx =  DirichletBC(W.sub(0),Constant(0.0),Bottom)
fby =  DirichletBC(W.sub(1),Constant(0.0),Bottom)
fbz = DirichletBC(W.sub(2), Constant(0.0), BottomL, method = 'pointwise')
bc_disp = [fty, fbx, fby, fbz ]


#--------------------------------------------------------
#               Define Variational Problem
#--------------------------------------------------------

# initialization 

u_old,u_conv, unew = Function(W),Function(W), Function(W)
phi_old,phi_conv = Function(V),Function(V)

E_du = ((1 - phi_old)**2 + 1e-6)*inner(grad(v),sigma(u))*dx

E_phi = ( Gc*l_o*inner(grad(p),grad(q))+\
            ((Gc/l_o) + 2.*en_dens(unew))*inner(p,q)-\
            2.*en_dens(unew)*q)*dx
u = Function(W)
p = Function(V)
p_disp = LinearVariationalProblem(lhs(E_du),rhs(E_du),u,bc_disp)
p_phi = LinearVariationalProblem(lhs(E_phi),rhs(E_phi),p)

# solver

solver_disp = LinearVariationalSolver(p_disp)
solver_phi = LinearVariationalSolver(p_phi)

# loading i.e. displacement

min_load = 0.00001
max_load = 0.07

t = 1e-4
deltaT = 1e-4


ut = 1
cvtk = Function(V)
dvtk = Function(W)
vtkplot = File('phi.pvd')
ptkplot = File('Displacement.pvd')
#------------------------------------- 
# displacement controlled test 
# starts here
#------------------------------------

while t <= max_load:
    u_Lx.t=t*ut
    
    u_old.assign(u_conv)
    phi_old.assign(phi_conv)
    iter = 1
    toll = 1e-2
    err = 1
    maxiter = 100
    while err > toll:
        solver_disp.solve()
        unew.assign(u)
        solver_phi.solve()
	
        err_u = errornorm(u,u_old, norm_type = 'l2', mesh = None)/norm(u)
        err_phi = errornorm(p,phi_old, norm_type = 'l2', mesh = None)/norm(p)
	
        err = max(err_u,err_phi)
	
        u_old.assign(u)
        phi_old.assign(p)


        iter = iter + 1
        if err < toll:
            print "Solution Converge after",iter
            u_conv.assign(u)
            phi_conv.assign(p)
            
            cvtk.assign(p)
            vtkplot << cvtk, t
            dvtk.assign(u)
            ptkplot << dvtk, t
            ux,uy, uz = split(u)
            plot(ux,key = 'ux',title = 'u_dispx')
            plot(uy,key ='uy',title = 'u_dispy')
            plot(uz, key = 'uz', title = 'u_dispz')
            plot(p,range_min = 0.,range_max = 1.,key = 'phi',title = 'phi%.5f'%(t))
                        

	    	    
    t+=deltaT

print 'solution done with no error :)'


