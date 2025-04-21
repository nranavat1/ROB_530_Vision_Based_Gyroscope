import cv2
import numpy as np
from scipy.linalg import expm
from filterpy.kalman import KalmanFilter
from tqdm import tqdm

class RGBDMotionEstimator:
    def __init__(self, flow_data, camera_intrinsics, use_ekf=False):
        self.flow_data = flow_data
        self.camera_intrinsics = camera_intrinsics
        self.use_ekf = use_ekf
        self.T_matrices = self.estimate_motion()

        if self.use_ekf:
            self.apply_ekf()

    def estimate_motion(self):
        T_matrices = []
        K_inv = np.linalg.inv(self.camera_intrinsics)

        progress_bar = tqdm(total=len(self.flow_data), desc="Estimating Motion", unit="frame")

        for flow, depth in self.flow_data:
            if flow is None or depth is None or np.all(flow == 0) or np.all(depth == 0):
                print("Skipping frame due to invalid flow or depth data.")
                progress_bar.update(1)
                continue

            h, w = flow.shape[:2]
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            pixel_coords = np.stack((x_coords, y_coords, np.ones_like(x_coords)), axis=-1).reshape(-1, 3)

            depth_values = depth.reshape(-1, 1)
            points_3d = (K_inv @ pixel_coords.T * depth_values.T).T

            flow_x = flow[..., 0].reshape(-1, 1)
            flow_y = flow[..., 1].reshape(-1, 1)
            moved_pixel_coords = pixel_coords + np.hstack((flow_x, flow_y, np.zeros_like(flow_x)))
            moved_points_3d = (K_inv @ moved_pixel_coords.T * depth_values.T).T

            points_2d = points_3d[:, :2].astype(np.float32)
            moved_points_2d = moved_points_3d[:, :2].astype(np.float32)

            if points_2d.shape[0] < 5 or moved_points_2d.shape[0] < 5:
                print("Not enough valid 2D points, skipping frame.")
                progress_bar.update(1)
                continue

            E, mask = cv2.findEssentialMat(points_2d, moved_points_2d, self.camera_intrinsics, method=cv2.RANSAC, prob=0.999, threshold=1.0)
            _, R, t, _ = cv2.recoverPose(E, points_2d, moved_points_2d, self.camera_intrinsics)

            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = t.ravel()
            T_matrices.append(T)

            progress_bar.update(1)

        progress_bar.close()
        return T_matrices

    def apply_ekf(self):
        X_est = np.eye(4)
        P = np.eye(6) * 0.1
        Q = np.eye(6) * 0.01
        R = np.eye(6) * 0.1
        H = np.eye(6)

        filtered_T = []

        for T in self.T_matrices:
            F = np.eye(6)
            X_pred = X_est @ T
            P = F @ P @ F.T + Q

            Z = np.hstack((T[:3, 3], self.rotation_to_vector(T[:3, :3])))
            Y = Z - np.hstack((X_pred[:3, 3], self.rotation_to_vector(X_pred[:3, :3])))
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)

            X_update = X_pred.copy()
            X_update[:3, 3] += K[:3] @ Y
            X_update[:3, :3] = self.vector_to_rotation(K[3:] @ Y) @ X_pred[:3, :3]
            P = (np.eye(6) - K @ H) @ P

            filtered_T.append(X_update)
            X_est = X_update

        self.T_matrices = filtered_T

    def rotation_to_vector(self, R):
        theta = np.arccos((np.trace(R) - 1) / 2)
        if np.sin(theta) == 0:
            return np.zeros(3)
        return (theta / (2 * np.sin(theta))) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])

    def vector_to_rotation(self, w):
        if np.linalg.norm(w) < 1e-6:
            return np.eye(3)
        W_hat = np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])
        return expm(W_hat)

    def get_se3_matrices(self):
        return self.T_matrices