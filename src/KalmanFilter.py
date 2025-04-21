import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, camera_intrinsics, process_noise=None, measurement_noise=None):
        """
        Kalman Filter for fusing motion estimation (6DoF) and optical flow.

        Args:
            initial_state: np.array of shape (6,) => [x, y, z, roll, pitch, yaw]
            process_noise: np.array of shape (6,6)
            measurement_noise: np.array of shape (6,6)
        """
        self.state = initial_state
        self.camera_intrinsics = camera_intrinsics
        self.P = np.eye(6) * 1e-3  # Initial uncertainty

        self.Q = process_noise if process_noise is not None else np.eye(6) * 1e-4
        self.R = measurement_noise if measurement_noise is not None else np.eye(6) * 1e-2

        self.H = np.eye(6)  # Measurement function
        self.state_history = [initial_state.copy()]

    def estimate_pose_delta_from_flow_and_depth(self, flow, depth_map):
        """
        Approximate pose delta from optical flow and depth map.
        
        Args:
            flow: np.array of shape (H, W, 2) => Optical flow for the frame
            depth_map: np.array of shape (H, W) => Depth map (from .npy file)
            fx: focal length in x direction (camera intrinsic)
            fy: focal length in y direction (camera intrinsic)
            
        Returns:
            np.array of shape (6,) => Pose change [dx, dy, dz, droll, dpitch, dyaw]
        """
        fx = self.camera_intrinsics[0][0]
        fy = self.camera_intrinsics[1][1]
        #flow is optical flow and depth
        flow_x = flow[0][..., 0]
        flow_y = flow[0][..., 1]
        
        # Compute depth for each pixel and average the flow
        depth = depth_map.flatten()
        flow_x = flow_x.flatten()
        flow_y = flow_y.flatten()

        # Compute the deltas for translation (assuming planar scene)
        delta_x = -np.mean(flow_x * depth / fx)
        delta_y = -np.mean(flow_y * depth / fy)
        delta_z = 0.0  # For simplicity, assuming the scene is mostly planar

        # For rotational data, we can assume 0 change if we don't have information for it
        delta_roll = 0
        delta_pitch = 0
        delta_yaw = 0

        return np.array([delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw])

    def step(self, motion_estimate, optical_flow, depth_map):
        """
        Process a new frame: update state using motion_estimate, optical_flow, and depth_map.
        
        Args:
            motion_estimate: np.array of shape (6,) => new motion estimate [dx, dy, dz, droll, dpitch, dyaw]
            optical_flow: np.array of shape (H, W, 2) => Optical flow for the frame
            depth_map: np.array of shape (H, W) => Depth map (from .npy file)
        """
        # Prediction step (based on motion estimate)
        delta_pose = motion_estimate - self.state
        state_pred = self.state + delta_pose
        P_pred = self.P + self.Q

        # Measurement update using optical flow and depth map
        flow_pose_delta = self.estimate_pose_delta_from_flow_and_depth(optical_flow, depth_map)
        z = self.state + flow_pose_delta

        # Kalman Gain
        y = z - state_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)

        # Update step
        self.state = state_pred + K @ y
        self.P = (np.eye(6) - K @ self.H) @ P_pred
        self.state_history.append(self.state.copy())

    def get_states(self):
        """
        Return the list of estimated states over time.
        """
        return np.array(self.state_history)
    
    def plot_3d(self, gt):
        """
        Plot the estimated states over time in a 3D space (x, y, z).
        """
        poses = self.get_states()
        # Extract positions (x, y, z) and orientation (roll, pitch, yaw)
        x = poses[:, 0]
        y = poses[:, 1]
        z = poses[:, 2]
        roll = poses[:, 3]
        pitch = poses[:, 4]
        yaw = poses[:, 5]

        # Plotting the trajectory and orientation vectors (using yaw)
        fig = plt.figure(figsize=(12, 6))

        # 3D plot of the trajectory
        ax = fig.add_subplot(121, projection='3d')
        ax.plot(x, y, z, label='Trajectory', color='b')
        ax.scatter(x[-1], y[-1], z[-1], color='r', label='Final Pose')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Trajectory from Pose Data')
        ax.legend()

        # 3D plot of the orientation as arrows
        ax2 = fig.add_subplot(122, projection='3d')
        ax2.plot(x, y, z, label='Trajectory', color='b')

        # Adding arrows to represent orientation at each pose
        for i in range(0, len(x), 10):  # Adjust step to change frequency of arrows
            dx = np.cos(yaw[i])  # x component of orientation (yaw)
            dy = np.sin(yaw[i])  # y component of orientation (yaw)
            ax2.quiver(x[i], y[i], z[i], dx, dy, 0, length=0.1, normalize=True, color='g')  # Arrows in the xy plane

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        ax2.set_title('Trajectory with Orientation Vectors')
        ax2.legend()

        plt.show()


def run_kalman_filtering(motion_vectors, dt=1.0):
    """
    Run Kalman Filtering on a list of 3D motion vectors (dx, dy, dz).
    
    Parameters:
        motion_vectors (List[np.ndarray]): List of 3D numpy arrays
        dt (float): Time step between frames

    Returns:
        np.ndarray: Array of filtered positions over time
    """
    kf = KalmanFilter(dt=dt)
    poses = []

    for motion_vector in motion_vectors:
        kf.predict()
        updated_state = kf.update(motion_vector)
        poses.append(updated_state[:3].flatten())  # Get x, y, z

    return np.array(poses)

def plot_trajectory(states, title="Kalman Filtered 6DoF Trajectory"):
    """
    Plot position and orientation over time from 6DoF state history.

    Args:
        states: np.array of shape (T, 6), where each row is [x, y, z, roll, pitch, yaw]
        title: Title of the plot
    """
    states = np.array(states)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(states[:, 0], label='x')
    axs[0].plot(states[:, 1], label='y')
    axs[0].plot(states[:, 2], label='z')
    axs[0].set_ylabel("Position")
    axs[0].legend()
    axs[0].set_title(title)

    axs[1].plot(states[:, 3], label='roll')
    axs[1].plot(states[:, 4], label='pitch')
    axs[1].plot(states[:, 5], label='yaw')
    axs[1].set_ylabel("Orientation")
    axs[1].legend()
    axs[1].set_xlabel("Frame")

    plt.tight_layout()
    plt.show()
