from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from IPython.display import clear_output
import time

import cv2
import numpy as np
import imageio
from IPython.display import display, Video, Image

class SE3Visualizer:
    def __init__(self, se3_matrices):
        self.se3_matrices = se3_matrices
        self.positions = self.extract_positions()

    def extract_positions(self):
        positions = [np.array([0, 0, 0])]
        for T in self.se3_matrices:
            positions.append(T[:3, 3])
        return np.array(positions)

    def visualize_gyroscope(self, save_video=True, save_gif=True):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        frames = []
        R_cumulative = np.eye(3)  # Initial rotation matrix (I)

        for i, T in enumerate(self.se3_matrices):
            clear_output(wait=True)
            ax.cla()

            R_t = T[:3, :3]  # current frame rotation matrix
            R_cumulative = R_t @ R_cumulative  # cumulative rotation matrix
            print(f"Frame {i} R_cumulative:\n", R_cumulative)

            # New coordinates
            ax.quiver(0, 0, 0, *R_cumulative[:, 0], color='r', label='X-axis')
            ax.quiver(0, 0, 0, *R_cumulative[:, 1], color='g', label='Y-axis')
            ax.quiver(0, 0, 0, *R_cumulative[:, 2], color='b', label='Z-axis')

            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_zlim([-1, 1])
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Cumulative Gyroscope at Frame {i}")

            ax.legend()
            plt.draw()
            plt.pause(0.05)  # Refresh

            # Save state
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frames.append(frame)

        print(f"Total frames in GIF: {len(frames)}")

        if save_gif and len(frames) > 1:
            imageio.mimsave('cumulative_gyroscope.gif', frames, fps=10)
            display(Image('cumulative_gyroscope.gif'))
        else:
            print("GIF Only have 1 frame")

        plt.close(fig)



    def visualize_trajectory(self, save_video=True, save_gif=True):
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection='3d')

        frames = []

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('trajectory.mp4', fourcc, 10, (500, 500))

        for i in range(1, len(self.positions)):
            ax.cla()
            ax.scatter(self.positions[:i, 0], self.positions[:i, 1], self.positions[:i, 2], color='b', marker='o')
            ax.plot(self.positions[:i, 0], self.positions[:i, 1], self.positions[:i, 2], color='r', linestyle='-')

            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title('3D Trajectory')

            plt.draw()
            fig.canvas.flush_events()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            if save_video:
                out.write(frame)

            frames.append(frame)

        if save_video:
            out.release()
            display(Video('trajectory.mp4'))

        if save_gif:
            imageio.mimsave('trajectory.gif', frames, fps=10)
            display(Image(filename='trajectory.gif'))

        plt.close(fig)
