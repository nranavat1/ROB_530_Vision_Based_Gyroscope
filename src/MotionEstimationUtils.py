import math
import numpy as np


# --- Helper function: Check if it's a valid rotation matrix ---
def isRotationMatrix(R):
    """ Checks if a matrix is a valid rotation matrix. """
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    # Set a small tolerance error threshold
    return n < 1e-5

# --- Helper function: Convert rotation matrix to Euler angles (RPY - ZYX order) ---
def rotationMatrixToEulerAngles(R):
    """ Calculates Euler angles (roll, pitch, yaw) corresponding to a rotation matrix
        Note: Assumes ZYX order (i.e., Yaw rotation around Z-axis first, then Pitch around Y-axis, and finally Roll around X-axis)
    """
    if not isRotationMatrix(R):
        print("Warning: Input is not a valid rotation matrix!")
        # Return a zero vector or handle the error as needed
        # return np.zeros(3)
        # Attempt to continue calculation, but the result may be meaningless
        pass

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6 # Check if close to gimbal lock

    if not singular: # Non-gimbal lock case
        x = math.atan2(R[2,1] , R[2,2]) # Roll (around X-axis)
        y = math.atan2(-R[2,0], sy)      # Pitch (around Y-axis)
        z = math.atan2(R[1,0], R[0,0]) # Yaw (around Z-axis)
    else: # Gimbal lock case
        # When Pitch approaches +/- 90 degrees, Roll and Yaw become coupled
        x = math.atan2(-R[1,2], R[1,1]) # Roll
        y = math.atan2(-R[2,0], sy)      # Pitch (+/- pi/2)
        z = 0                          # Yaw (cannot be uniquely determined, set to 0)

    # Return value is in radians: roll (x), pitch (y), yaw (z)
    return np.array([x, y, z])

# --- Helper function: Calculate the inverse of the SE(3) transformation matrix ---
def inverse_se3(T):
    """ Calculate the inverse of the 4x4 SE(3) matrix T = [[R, t],[0, 1]] """
    if T.shape != (4, 4):
        raise ValueError("Input must be a 4x4 matrix")
    R = T[0:3, 0:3]
    t = T[0:3, 3]
    # Check if R is close to a rotation matrix (optional but recommended)
    # if not isRotationMatrix(R):
    #     print("Warning: The rotation part of the SE(3) matrix may be invalid.")
    R_inv = R.T
    t_inv = -R_inv @ t
    T_inv = np.identity(4, dtype=T.dtype) # Keep data type consistent
    T_inv[0:3, 0:3] = R_inv
    T_inv[0:3, 3] = t_inv
    return T_inv