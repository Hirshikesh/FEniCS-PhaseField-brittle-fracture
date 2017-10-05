import numpy as np

def D_Matrix(E, nu, stressState):

    if stressState=='PlaneStress':
        a = E/(1-nu**2)
        D = np.array([[a, a*nu, 0],
                      [a*nu, a, 0],
                      [0, 0, a*(1-nu)/2],
                      ])
    else:
        a = E/((1+nu)*(1-2*nu))
        D = np.array([[a*(1-nu), a*nu, 0],
                      [a*nu, a*(1-nu), 0],
                      [0, 0, a*(0.5-nu)],
                      ])
    return D
