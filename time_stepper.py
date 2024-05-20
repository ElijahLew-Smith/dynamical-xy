
import numpy as np



def zero_force(lattice, boundary_conditions): 
    return np.zeros_like(lattice)


    ###  ------------------ Force Functions ------------------
def shift(lattice, x, y):
    shifted_lattice = np.roll(np.roll(lattice, shift=x, axis=0), shift=y, axis=1)
    return shifted_lattice

def dissipative_force(lattice, boundary_conditions):
    f1 = np.sin(shift(lattice, 1, 0) - lattice)
    f2 = np.sin(shift(lattice,-1, 0) - lattice)
    f3 = np.sin(shift(lattice, 0, 1) - lattice)
    f4 = np.sin(shift(lattice, 0,-1) - lattice)
    if boundary_conditions == 'open':
        f1[ 0,:] = 0
        f2[-1,:] = 0
        f3[:, 0] = 0
        f4[:,-1] = 0
    force = f1 + f2 + f3 + f4
    return force

def kuramoto_dV(lattice, boundary_conditions):
    f1 = np.cos(shift(lattice, 1, 0)-lattice)
    f2 = np.cos(shift(lattice,-1, 0)-lattice)
    f3 = np.cos(shift(lattice, 0, 1)-lattice)
    f4 = np.cos(shift(lattice, 0,-1)-lattice)
    force =  (f1 + f2 + f3 + f4)
    return force

def kuramoto_muV(lattice, boundary_conditions):
    g1 = -np.cos(shift(lattice, 1, 0)-lattice) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 2, 0)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 0, 0)))
    g2 =  np.cos(shift(lattice,-1, 0)-lattice) * (np.sin(shift(lattice,-1, 0)-shift(lattice, 0, 0)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-2, 0)))
    g3 = -np.cos(shift(lattice, 1, 0)-lattice) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 1,-1)))
    g4 =  np.cos(shift(lattice,-1, 0)-lattice) * (np.sin(shift(lattice,-1, 0)-shift(lattice,-1, 1)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-1,-1)))
    g5 = -np.cos(shift(lattice, 0, 1)-lattice) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 0, 1)-shift(lattice,-1, 1)))
    g6 =  np.cos(shift(lattice, 0,-1)-lattice) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 1,-1)) + np.sin(shift(lattice, 0,-1)-shift(lattice,-1,-1)))
    g7 = -np.cos(shift(lattice, 0, 1)-lattice) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 2)) + np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 0)))
    g8 =  np.cos(shift(lattice, 0,-1)-lattice) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 0, 0)) + np.sin(shift(lattice, 0,-1)-shift(lattice, 0,-2)))
    force = (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8)
    return force

def andy_dV1(lattice, boundary_conditions):
    f1 =  np.sin(shift(lattice, 1, 0)-lattice) * np.cos((shift(lattice, 1, 0)+lattice)/2)
    f2 = -np.sin(shift(lattice,-1, 0)-lattice) * np.cos((shift(lattice,-1, 0)+lattice)/2)
    f3 =  np.sin(shift(lattice, 0, 1)-lattice) * np.sin((shift(lattice, 0, 1)+lattice)/2)
    f4 = -np.sin(shift(lattice, 0,-1)-lattice) * np.sin((shift(lattice, 0,-1)+lattice)/2)
    if boundary_conditions == 'open':
        f1[ 0,:] = 0
        f2[-1,:] = 0
        f3[:, 0] = 0
        f4[:,-1] = 0
    force = .5 * (f1 + f2 + f3 + f4)
    return force

def andy_dV3(lattice, boundary_conditions):
    f1 =  (np.cos(shift(lattice, 1, 0)-lattice)-1) * np.sin((shift(lattice, 1, 0)+lattice)/2)
    f2 = -(np.cos(shift(lattice,-1, 0)-lattice)-1) * np.sin((shift(lattice,-1, 0)+lattice)/2)
    f3 = -(np.cos(shift(lattice, 0, 1)-lattice)-1) * np.cos((shift(lattice, 0, 1)+lattice)/2)
    f4 =  (np.cos(shift(lattice, 0,-1)-lattice)-1) * np.cos((shift(lattice, 0,-1)+lattice)/2)
    if boundary_conditions == 'open':
        f1[ 0,:] = 0
        f2[-1,:] = 0
        f3[:, 0] = 0
        f4[:,-1] = 0
    force = .25 *  (f1 + f2 + f3 + f4)
    return force

def andy_dV(lattice):
    return andy_dV1(lattice) + andy_dV3(lattice)

def andy_muV(lattice, boundary_conditions):
    f1 =  (np.cos(shift(lattice, 1, 0)-lattice)-1) * np.cos((shift(lattice, 1, 0)+lattice)/2) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 2, 0)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 0, 0)))
    f2 = -(np.cos(shift(lattice,-1, 0)-lattice)-1) * np.cos((shift(lattice,-1, 0)+lattice)/2) * (np.sin(shift(lattice,-1, 0)-shift(lattice, 0, 0)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-2, 0)))
    f3 =  (np.cos(shift(lattice, 1, 0)-lattice)-1) * np.cos((shift(lattice, 1, 0)+lattice)/2) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 1,-1)))
    f4 = -(np.cos(shift(lattice,-1, 0)-lattice)-1) * np.cos((shift(lattice,-1, 0)+lattice)/2) * (np.sin(shift(lattice,-1, 0)-shift(lattice,-1, 1)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-1,-1)))
    f5 =  (np.cos(shift(lattice, 0, 1)-lattice)-1) * np.sin((shift(lattice, 0, 1)+lattice)/2) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 0, 1)-shift(lattice,-1, 1)))
    f6 = -(np.cos(shift(lattice, 0,-1)-lattice)-1) * np.sin((shift(lattice, 0,-1)+lattice)/2) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 1,-1)) + np.sin(shift(lattice, 0,-1)-shift(lattice,-1,-1)))
    f7 =  (np.cos(shift(lattice, 0, 1)-lattice)-1) * np.sin((shift(lattice, 0, 1)+lattice)/2) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 2)) + np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 0)))
    f8 = -(np.cos(shift(lattice, 0,-1)-lattice)-1) * np.sin((shift(lattice, 0,-1)+lattice)/2) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 0, 0)) + np.sin(shift(lattice, 0,-1)-shift(lattice, 0,-2)))
    if boundary_conditions == 'open':
        f1[:2, :] = 0
        f2[-2:, :] = 0
        f3[ 0, :] = 0
        f4[-1, :] = 0
        f5[:,  0] = 0
        f6[:, -1] = 0
        f3[:, [0,-1]] = 0
        f4[:, [0,-1]] = 0
        f5[[0,-1], :] = 0
        f6[[0,-1], :] = 0
        f7[:, :2] = 0
        f8[:, -2:] = 0
    force = .5 * (f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8)
    return force

def marvin_dV(lattice, boundary_conditions):
    f1 =  np.sin(shift(lattice, 1, 0))
    f2 = -np.sin(shift(lattice,-1, 0))
    f3 = -np.cos(shift(lattice, 0, 1))
    f4 =  np.cos(shift(lattice, 0,-1))
    if boundary_conditions == 'open':
        f1[ 0,:] = 0
        f2[-1,:] = 0
        f3[:, 0] = 0
        f4[:,-1] = 0
    force = .5 * (f1 + f2 + f3 + f4)
    return force

def marvin_muV(lattice, boundary_conditions):
    g1 =  (np.cos(lattice) + np.cos(shift(lattice, 1, 0))) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 2, 0)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 0, 0)))
    g2 = -(np.cos(lattice) + np.cos(shift(lattice,-1, 0))) * (np.sin(shift(lattice,-1, 0)-shift(lattice, 0, 0)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-2, 0)))
    g3 =  (np.cos(lattice) + np.cos(shift(lattice, 1, 0))) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 1,-1)))
    g4 = -(np.cos(lattice) + np.cos(shift(lattice,-1, 0))) * (np.sin(shift(lattice,-1, 0)-shift(lattice,-1, 1)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-1,-1)))
    g5 =  (np.sin(lattice) + np.sin(shift(lattice, 0, 1))) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 0, 1)-shift(lattice,-1, 1)))
    g6 = -(np.sin(lattice) + np.sin(shift(lattice, 0,-1))) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 1,-1)) + np.sin(shift(lattice, 0,-1)-shift(lattice,-1,-1)))
    g7 =  (np.sin(lattice) + np.sin(shift(lattice, 0, 1))) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 2)) + np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 0)))
    g8 = -(np.sin(lattice) + np.sin(shift(lattice, 0,-1))) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 0, 0)) + np.sin(shift(lattice, 0,-1)-shift(lattice, 0,-2)))
    if boundary_conditions == 'open':
        g1[:1, :] = 0
        g2[-2:, :] = 0
        g3[ 0, :] = 0
        g4[-1, :] = 0
        g5[:,  0] = 0
        g6[:, -1] = 0
        g3[:, [0,-1]] = 0
        g4[:, [0,-1]] = 0
        g5[[0,-1], :] = 0
        g6[[0,-1], :] = 0
        g7[:, :1] = 0
        g8[:, -2:] = 0
    force = .5 * (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8)
    return force

def eli_dV(lattice, boundary_conditions):
    f1 =  np.sin((lattice + shift(lattice, 1, 0))/2)
    f2 = -np.sin((lattice + shift(lattice,-1, 0))/2)
    f3 = -np.cos((lattice + shift(lattice, 0, 1))/2)
    f4 =  np.cos((lattice + shift(lattice, 0,-1))/2)
    if boundary_conditions == 'open':
        f1[ 0,:] = 0
        f2[-1,:] = 0
        f3[:, 0] = 0
        f4[:,-1] = 0
    force =  (f1 + f2 + f3 + f4)
    return force

def eli_muV(lattice, boundary_conditions):
    g1 =  (np.cos(.5*(lattice + shift(lattice, 1, 0)))) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 2, 0)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 0, 0)))
    g2 = -(np.cos(.5*(lattice + shift(lattice,-1, 0)))) * (np.sin(shift(lattice,-1, 0)-shift(lattice, 0, 0)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-2, 0)))
    g3 =  (np.cos(.5*(lattice + shift(lattice, 1, 0)))) * (np.sin(shift(lattice, 1, 0)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 1, 0)-shift(lattice, 1,-1)))
    g4 = -(np.cos(.5*(lattice + shift(lattice,-1, 0)))) * (np.sin(shift(lattice,-1, 0)-shift(lattice,-1, 1)) + np.sin(shift(lattice,-1, 0)-shift(lattice,-1,-1)))
    g5 =  (np.sin(.5*(lattice + shift(lattice, 0, 1)))) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 1, 1)) + np.sin(shift(lattice, 0, 1)-shift(lattice,-1, 1)))
    g6 = -(np.sin(.5*(lattice + shift(lattice, 0,-1)))) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 1,-1)) + np.sin(shift(lattice, 0,-1)-shift(lattice,-1,-1)))
    g7 =  (np.sin(.5*(lattice + shift(lattice, 0, 1)))) * (np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 2)) + np.sin(shift(lattice, 0, 1)-shift(lattice, 0, 0)))
    g8 = -(np.sin(.5*(lattice + shift(lattice, 0,-1)))) * (np.sin(shift(lattice, 0,-1)-shift(lattice, 0, 0)) + np.sin(shift(lattice, 0,-1)-shift(lattice, 0,-2)))
    if boundary_conditions == 'open':
        g1[:1, :] = 0
        g2[-2:, :] = 0
        g3[ 0, :] = 0
        g4[-1, :] = 0
        g5[:,  0] = 0
        g6[:, -1] = 0
        g3[:, [0,-1]] = 0
        g4[:, [0,-1]] = 0
        g5[[0,-1], :] = 0
        g6[[0,-1], :] = 0
        g7[:, :1] = 0
        g8[:, -2:] = 0
    force = 2 * (g1 + g2 + g3 + g4 + g5 + g6 + g7 + g8)
    return force


###------------------ Force Initialization ------------------




# def initialize_force(model_type, active_component, muV_function, boundary_conditions):

#     def create_force_function(dV_function):
#         def force_function(lattice, damping, temp, activity):
#            total_force = dissipative_force(lattice, boundary_conditions) * damping
#            if 'dV' in active_component:
#                total_force += dV_function(lattice, boundary_conditions) * activity
#            if 'muV' in active_component:
#                total_force += muV_function(lattice, boundary_conditions) * activity / temp
#            return total_force
#         return force_function

#     if model_type == 'XY':
#         force_function = create_force_function(zero_force, zero_force)
#     elif model_type == 'kuramoto':
#         force_function = create_force_function(kuramoto_dV, kuramoto_muV)
#     elif model_type == '1':
#         force_function = create_force_function(andy_dV, andy_muV)
#     elif model_type == '2':
#         force_function = create_force_function(marvin_dV, marvin_muV)
#     elif model_type == '3':
#         force_function = create_force_function(eli_dV, eli_muV)
#     else:
#         raise ValueError("Invalid model type")
    
#     return force_function


# def total_force(lattice, damping, temp, activity, full_force_function):
#     total_force = full_force_function(lattice, damping, temp, activity)
#     return total_force

def create_force_function(active_component, dV_function, muV_function):
    """
    This function takes in what the dV and muV forces are and spits out a function that will include exactly those. This is the function we run time evolution on. 
    Its initialized this way so that one doesn't have to decide what force to use in every timestep. 
    """
    def total_force(lattice, damping, temp, activity, boundary_conditions):
        force = np.zeros_like(lattice)
        if damping != 0:
            force =  dissipative_force(lattice, boundary_conditions) * damping
        if 'dV' in active_component:
            force += dV_function(lattice, boundary_conditions) * activity
        if 'muV' in active_component and temp != 0:
            force += (muV_function(lattice, boundary_conditions) / temp) * activity 
        return force
    return total_force

def initialize_force(model_type, active_component):
    """
    Sets the value of the total force function by running "create_force_function" with the appropriate parameters.
    """
    if model_type == 'XY':
        #This will just give for the active parts 
        total_force = create_force_function(active_component, zero_force, zero_force)
    elif model_type == 'kuramoto':
        total_force = create_force_function(active_component, kuramoto_dV, kuramoto_muV)
    elif model_type == '1':
        total_force = create_force_function(active_component, andy_dV, andy_muV)
    elif model_type == '2':
        total_force = create_force_function(active_component, marvin_dV, marvin_muV)
    elif model_type == '3':
        total_force = create_force_function(active_component, eli_dV, eli_muV)
    else:
        raise ValueError(f"Model type {model_type} is not valid")
    return total_force


def runge_kutta_step(lattice, total_force, damping, noise, temp, activity, dt, boundary_conditions):
    Y0 = lattice

    zeta_1= np.random.standard_normal(Y0.shape)
    xi    = np.random.standard_normal(Y0.shape)
    mu    = np.random.standard_normal(Y0.shape)
    eta   = np.random.standard_normal(Y0.shape)
    phi   = np.random.standard_normal(Y0.shape)

    #since p=1 we can directly input the values of ρ and α
    a_10 = - 1/np.pi * np.sqrt(2 * dt) * xi - 2 * np.sqrt(dt * (1/12 - 1/(2 * np.pi**2))) * mu
    b_1  = np.sqrt(dt/2) * eta + np.sqrt(dt * ((np.pi**2 / 180) - (1 / (2 * np.pi**2)))) * phi

    #the final term of J vanishes for p=1
    dW = np.sqrt(dt) * zeta_1
    dZ = .5 * dt * (np.sqrt(dt) * zeta_1 + a_10)
    J_110 =  (1/6) * dt**2 * zeta_1**2 + .25 * dt * a_10**2 - (.5/np.pi) * dt**(1.5) * zeta_1 * b_1 + .25 * dt**(1.5) * a_10 * zeta_1
    argument = np.abs(2 * J_110 * dt - dZ**2)#otherwise the floating point arithmetic can't take the square root
    Ybar1 = Y0 + .5 * total_force(Y0, damping, temp, activity, boundary_conditions) * dt + (noise / dt) * (dZ + np.sqrt(argument))
    Ybar2 = Y0 + .5 * total_force(Y0, damping, temp, activity, boundary_conditions) * dt + (noise / dt) * (dZ - np.sqrt(argument))

    Y1 = Y0 + .5 * (total_force(Ybar1, damping, temp, activity, boundary_conditions) + total_force(Ybar2, damping, temp, activity, boundary_conditions)) * dt + noise * dW
    return Y1
