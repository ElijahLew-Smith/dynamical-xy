
import numpy as np 
from scipy.ndimage.filters import gaussian_filter
from scipy.stats import circmean

###  ------------------ Initialization  ------------------
def _perpendicular_vector_angle(vector):
    x, y = vector
    perpendicular_x, perpendicular_y = (-y, x)
    angle_rad = np.arctan2(perpendicular_y, perpendicular_x)
    if angle_rad < 0:
        angle_rad += 2 * np.pi
    return angle_rad

def initial_condition(X, Y, initial_background, background_angle, perturbation_type, perturbation_size, perturbation_angle):
    center1 = (X // 2, Y // 4)
    center2 = (X // 2, 3* Y // 4)
    center = [X//2,Y//2]
    if initial_background == 'uniform':
        theta_initial = np.zeros((X, Y)) + background_angle
    elif initial_background == 'random':
        theta_initial = np.random.rand(X, Y) * np.pi * 2

    if perturbation_type == 'none':
        pass
    elif perturbation_type == 'smooth_random':
        theta_random = np.random.rand(X, Y) * np.pi * 2
        theta_temporary = np.copy(theta_random)
        filter_radius = 10
        grid_x, grid_y = np.meshgrid(np.arange(Y), np.arange(X))
        for i in range(X):
            for j in range(Y):
                distances = np.sqrt((grid_x - i)**2 + (grid_y - j)**2)
                circular_indices = np.where(distances <= filter_radius)
                subarray = theta_temporary[circular_indices]
                theta_random[i, j] = circmean(subarray)
    elif perturbation_type == 'random_blocks':
        for k in range(X // perturbation_size):
            for l in range(Y // perturbation_size):
                random_number = np.random.rand()*np.pi*2
                for i in range(perturbation_size):
                    for j in range(perturbation_size):
                        theta_initial[i+k*perturbation_size, j+l*perturbation_size] = random_number
    elif perturbation_type == 'stripe':
        theta_initial[X//2 - perturbation_size//2:X//2 + perturbation_size//2,:]=perturbation_angle
        theta_initial = gaussian_filter(theta_initial, sigma=perturbation_size/5)
    elif perturbation_type == 'cross':
        theta_initial[X//2 - perturbation_size//2:X//2 + perturbation_size//2,:]=perturbation_angle
        theta_initial[:,Y//2 - perturbation_size//2:Y//2 + perturbation_size//2]=perturbation_angle
        theta_initial[X//2 - perturbation_size//2:X//2 + perturbation_size//2,Y//2 - perturbation_size//2:Y//2 + perturbation_size//2]=background_angle-perturbation_angle
    elif perturbation_type == 'bottom':
        theta_initial[0:20,:] = background_angle+np.pi
    elif perturbation_type == 'half_random':
        theta_initial[:, Y//4:]
        for i in range(int(X)):
            for j in range(int(Y/4), int(3*Y/4)):
                theta_initial[i, j] = np.pi+background_angle
    elif perturbation_type == 'uniform':
        theta_initial = theta_initial
    elif perturbation_type == 'blob':
        for i in range(X):
            for j in range(Y):
                x, y = j - center[0], i - center[1]
                radius = np.sqrt(x**2 + y**2)
                if radius <= perturbation_size / 2:
                    theta_initial[i,j] = perturbation_angle
        theta_initial = gaussian_filter(theta_initial, sigma=perturbation_size/5)

    elif perturbation_type == 'blob_random':
        for i in range(X):
            for j in range(Y):
                x, y = j - center[0], i - center[1]
                radius = np.sqrt(x**2 + y**2)
                if radius <= perturbation_size / 2:
                    theta_initial[i, j] = np.random.rand() * np.pi * 2
    elif perturbation_type == 'vortex':
        for i in range(X):
            for j in range(Y):
                theta_initial[i, j] = (np.arctan2(j - center[1], i - center[0])+perturbation_angle )% (2 * np.pi)

    elif perturbation_type == 'anti_vortex':
        for i in range(X):
            for j in range(Y):
                theta_initial[i, j] = (np.arctan2(i - center[0], j - center[1])+perturbation_angle )% (2 * np.pi)

    elif perturbation_type == 'vorticies':
        for i in range(X):
            for j in range(Y):
                center1 = [center[0]-perturbation_size/2, center[1]]
                center2 = [center[0]+perturbation_size/2, center[1]]
                votex1 = np.arctan2(i - center1[0], j - center1[1])
                vortex2 = np.arctan2(i - center2[0], j - center2[1])
                theta_initial[i, j] = (votex1-vortex2+perturbation_angle)% (2 * np.pi)
    elif perturbation_type == 'same_vorticies':
        for i in range(X):
            for j in range(Y):
                center1 = [center[0]-perturbation_size, center[1]]
                center2 = [center[0]+perturbation_size, center[1]]
                votex1 = np.arctan2(i - center1[0], j - center1[1])
                vortex2 = np.arctan2(i - center2[0], j - center2[1])
                theta_initial[i, j] = (votex1+vortex2+perturbation_angle)% (2 * np.pi)
    elif perturbation_type == 'perturbation':
        if perturbation_size == 1:
            theta_initial[center[0], center[1]] = perturbation_angle
        else:
            for i in range(center[0] - perturbation_size//2, center[0] + perturbation_size//2):
                for j in range(center[1] - perturbation_size//2, center[1] + perturbation_size//2):
                    theta_initial[i, j] = perturbation_angle
    else:
        raise ValueError("Invalid initial perturbation type")
    
    lattice = theta_initial % (np.pi * 2)

    return lattice 