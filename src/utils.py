import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import cv2
from ipywidgets import interact, IntSlider

def dict_to_list(traj):
    final = []
    for x in traj:
      traj_list = []
      for i, j in x.items():
        traj_list.append(j)
      final.append(traj_list)
    return np.array(final)

def plot_3d_traj_orientation(trajectory):
        # Extract positions (x, y, z) and orientation (roll, pitch, yaw)

        poses = np.array(dict_to_list(trajectory))
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

def plot_3d_traj(traj, gt):
      """
      Plot the estimated states over time in a 3D space (x, y, z).
      """
      states = np.array(dict_to_list(traj))
      x = states[:, 0]
      y = states[:, 1]
      z = states[:, 2]

      fig = plt.figure(figsize=(14, 6))
      ax = fig.add_subplot(121, projection='3d')
      ax.plot(x, y, z, label='Estimated Path', color='b')
      ax.scatter(x[-1], y[-1], z[-1], color='r', label='Final State')

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.set_title('Kalman Filter Estimated 3D Path')
      ax.legend()


      ax = fig.add_subplot(122, projection='3d')
      ax.plot(gt[0], gt[1], gt[2], label='Ground Truth Path', color='b')
      ax.scatter(gt[0][-1], gt[1][-1], gt[2][-1], color='r', label='Final State')

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.set_title('Kalman Filter Ground Truth Path')
      ax.legend()

      plt.show()

def load_pose_file(filename):
    """
    Load a pose file with columns:
    timestamp tx ty tz qx qy qz qw

    Args:
        filename: path to the text file
        as_int: whether to convert values to integers (default: False = floats)

    Returns:
        np.ndarray of shape (N, 8)
    """
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Skip header if present
    if lines[0].strip().startswith("timestamp"):
        lines = lines[1:]

    data = []
    for line in lines[3:]:
        values = line.strip().split()
        values = [float(v) for v in values]
        data.append(values)

    return np.array(data)

def associate_rgb_depth(rgb_file, depth_file, output_file, max_difference=0.02):
    rgb_data = np.loadtxt(rgb_file, dtype=str, comments="#")
    depth_data = np.loadtxt(depth_file, dtype=str, comments="#")

    rgb_timestamps = rgb_data[:, 0].astype(float)
    depth_timestamps = depth_data[:, 0].astype(float)

    associations = []

    for rgb_time, rgb_name in zip(rgb_timestamps, rgb_data[:, 1]):
        index = np.argmin(np.abs(depth_timestamps - rgb_time))
        time_diff = abs(depth_timestamps[index] - rgb_time)

        if time_diff < max_difference:
            associations.append(f"{rgb_time} {rgb_name} {depth_timestamps[index]} {depth_data[index, 1]}")

    with open(output_file, "w") as f:
        f.write("\n".join(associations))

    print(f"✅ Associated timestamps saved to {output_file}")



def visualize_single_flow(flow, step=15):
    """
    Generate an RGB image of flow vectors for a single frame.

    Parameters:
        flow: Optical flow array
        step: Sampling step size
    Returns:
        RGB visualization as a numpy array
    """
    h, w = flow.shape[:2]
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(0, h, step):
        for x in range(0, w, step):
            fx, fy = flow[y, x]
            mag = np.sqrt(fx ** 2 + fy ** 2)
            if mag > 1:
                end_x, end_y = int(x + fx), int(y + fy)
                color = (0, int(min(255, mag * 5)), int(min(255, 255 - mag * 5)))
                vis = cv2.arrowedLine(vis, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
    return vis

def interactive_flow_viewer(flow_data, step=1):
    """
    Visualize optical flow frames using a slider for interaction.

    Parameters:
        flow_data: list of (flow, depth) tuples
        step: number of frames to skip per slider increment
    """
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.2)

    initial_flow = flow_data[0][0]
    vis = visualize_single_flow(initial_flow)
    im = ax.imshow(vis)
    ax.set_title("Optical Flow (Use slider below)")
    ax.axis("off")

    # Slider setup
    ax_slider = plt.axes([0.2, 0.05, 0.65, 0.03])
    slider = Slider(ax_slider, "Frame", 0, len(flow_data)-1, valinit=0, valstep=step)

    def update(val):
        frame_idx = int(slider.val)
        flow, _ = flow_data[frame_idx]
        vis = visualize_single_flow(flow)
        im.set_data(vis)  # Update the image data
        fig.canvas.draw_idle()  # Redraw the figure to reflect changes

    slider.on_changed(update)
    plt.show()
