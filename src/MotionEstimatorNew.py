import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

class RGBDMotionEstimator:
    def __init__(self, flow_data, camera_intrinsics, use_ekf=False):
        self.flow_data = flow_data
        self.K = camera_intrinsics
        self.use_ekf = use_ekf
        self.trajectory = self.estimate_motion()

        if self.use_ekf:
            self.apply_ekf()

    def estimate_motion(self):
        poses = []
        K_inv = np.linalg.inv(self.K)
        current_pose = np.eye(4)  # Start at origin

        progress_bar = tqdm(total=len(self.flow_data), desc="Estimating Motion", unit="frame",leave=False )

        for flow, depth in self.flow_data:
            if flow is None or depth is None or np.all(flow == 0) or np.all(depth == 0):
                print("Skipping frame due to invalid flow or depth data.")
                progress_bar.update(1)
                continue

            h, w = flow.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            pixel_coords = np.stack((x_coords, y_coords, np.ones_like(x_coords)), axis=-1).reshape(-1, 3)

            depth_values = depth.reshape(-1, 1)
            valid = depth_values > 0
            pixel_coords = pixel_coords[valid.flatten()]
            depth_values = depth_values[valid.flatten()]

            points_3d = (K_inv @ pixel_coords.T * depth_values.T).T

            flow_x = flow[..., 0].reshape(-1, 1)[valid.flatten()]
            flow_y = flow[..., 1].reshape(-1, 1)[valid.flatten()]
            moved_pixel_coords = pixel_coords + np.hstack((flow_x, flow_y, np.zeros_like(flow_x)))
            moved_points_3d = (K_inv @ moved_pixel_coords.T * depth_values.T).T

            points_2d = points_3d[:, :2].astype(np.float32)
            moved_points_2d = moved_points_3d[:, :2].astype(np.float32)

            if points_2d.shape[0] < 5:
                print("Not enough valid 2D points, skipping frame.")
                progress_bar.update(1)
                continue

            E, mask = cv2.findEssentialMat(points_2d, moved_points_2d, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R_mat, t, _ = cv2.recoverPose(E, points_2d, moved_points_2d, self.K)

            T = np.eye(4)
            T[:3, :3] = R_mat
            T[:3, 3] = t.ravel()

            # Update current pose
            current_pose = current_pose @ np.linalg.inv(T)
            poses.append(self.pose_to_xyzrpy(current_pose))

            progress_bar.update(1)

        progress_bar.close()
        return poses

    def pose_to_xyzrpy(self, T):
        x, y, z = T[:3, 3]
        rotation = R.from_matrix(T[:3, :3])
        roll, pitch, yaw = rotation.as_euler('xyz', degrees=False)
        return {
            'x': x, 'y': y, 'z': z,
            'roll': roll, 'pitch': pitch, 'yaw': yaw
        }

    def apply_ekf(self):
        # Placeholder for EKF integration
        print("EKF filtering not yet implemented.")

    def get_trajectory(self):
        return self.trajectory