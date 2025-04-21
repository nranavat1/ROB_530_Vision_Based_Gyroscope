import numpy as np
import cv2
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt

class TUMRGBDOpticalFlowTracker:
    def __init__(self, rgb_folder, depth_folder, timestamps_file, model="lk", visualization=True):
        self.rgb_folder = rgb_folder
        self.depth_folder = depth_folder
        self.timestamps = self.load_timestamps(timestamps_file)
        self.model = model.lower()
        self.visualization = visualization

        self.frame_idx = 0
        self.old_rgb = None
        self.old_depth = None
        self.p0 = None

        if self.model == "lk":
            self.feature_params = dict(maxCorners=500, qualityLevel=0.3, minDistance=7, blockSize=7)
            self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                                  criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        elif self.model == "farneback":
            pass

    def load_timestamps(self, timestamps_file):
        timestamps = []
        with open(timestamps_file, 'r') as f:
            for line in f.readlines():
                if line.strip() and not line.startswith("#"):
                    values = line.strip().split()
                    if len(values) >= 2:
                        timestamps.append((values[1], values[3]))
        return timestamps

    def load_frame(self, idx):
        if idx >= len(self.timestamps):
            return None, None

        rgb_filename, depth_filename = self.timestamps[idx]

        rgb_filename = rgb_filename.split("/")[-1]
        depth_filename = depth_filename.split("/")[-1]

        rgb_path = os.path.join(self.rgb_folder, rgb_filename)
        depth_path = os.path.join(self.depth_folder, depth_filename)

        # print(f"Loading RGB: {rgb_path}, Depth: {depth_path}")

        rgb_frame = cv2.imread(rgb_path)
        depth_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)

        if rgb_frame is None or depth_frame is None:
            print(f"Error: Cannot read RGB: {rgb_path} or Depth: {depth_path}")
            return None, None
        return rgb_frame, depth_frame

    def initialize(self):
        self.old_rgb, self.old_depth = self.load_frame(self.frame_idx)

        if self.old_rgb is None or self.old_depth is None:
            print("Error: Cannot read RGB or Depth images.")
            return False

        self.old_gray = cv2.cvtColor(self.old_rgb, cv2.COLOR_BGR2GRAY)
        if self.model == "lk":
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)

        return True

    def process(self):
        if not self.initialize():
            return []

        flow_data = []
        total_frames = len(self.timestamps)
        progress_bar = tqdm(total=total_frames, desc="Processing TUM RGBD Data", unit="frame", leave=False)

        for self.frame_idx in range(1, total_frames):
            rgb_frame, depth_frame = self.load_frame(self.frame_idx)
            if rgb_frame is None or depth_frame is None:
                break

            frame_gray = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2GRAY)

            if self.model == "lk":
                flow = self.lk_optical_flow(frame_gray)
            elif self.model == "farneback":
                flow = self.farneback_optical_flow(frame_gray)
            else:
                flow = None

            flow_data.append((flow, depth_frame))

            if self.visualization and flow is not None:
                self.visualize_optical_flow(flow)

            progress_bar.update(1)

        progress_bar.close()
        
        if self.visualization:
          cv2.destroyAllWindows()

        return flow_data

    def lk_optical_flow(self, frame_gray):
        if self.p0 is None or len(self.p0) == 0:
            self.p0 = cv2.goodFeaturesToTrack(self.old_gray, mask=None, **self.feature_params)
        p1, st, err = cv2.calcOpticalFlowPyrLK(self.old_gray, frame_gray, self.p0, None, **self.lk_params)
        flow = np.zeros_like(frame_gray, dtype=np.float32)
        if p1 is not None:
            good_new = p1[st == 1]
            good_old = self.p0[st == 1]
            for new, old in zip(good_new, good_old):
                a, b = new.ravel()
                c, d = old.ravel()
                flow[int(d), int(c)] = [a - c, b - d]
            self.p0 = good_new.reshape(-1, 1, 2)
        self.old_gray = frame_gray.copy()
        return flow

    def farneback_optical_flow(self, frame_gray):
        flow = cv2.calcOpticalFlowFarneback(self.old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.old_gray = frame_gray.copy()
        return flow

    def visualize_optical_flow(self, flow):
        h, w = flow.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        step = 15
        for y in range(0, h, step):
            for x in range(0, w, step):
                fx, fy = flow[y, x]
                mag = np.sqrt(fx**2 + fy**2)
                if mag > 1:
                    end_x, end_y = int(x + fx), int(y + fy)
                    color = (0, int(min(255, mag * 5)), int(min(255, 255 - mag * 5)))
                    cv2.arrowedLine(vis, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        return vis

    def show_visualizations_in_grid(self, images, start_idx):
        n = len(images)
        cols = 5
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
        for i in range(rows * cols):
            ax = axes[i // cols, i % cols] if rows > 1 else axes[i % cols]
            if i < n:
                ax.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
                ax.set_title(f"Frame {start_idx + i * 10}")
            ax.axis("off")
        plt.tight_layout()
        plt.show()



def run_tum_rgbd_demo(rgb_folder, depth_folder, timestamps_file):
    # rgb_folder =   # RGB File
    # depth_folder =   # Depth File
    # timestamps_file =   # TUM Timestamp file

    if not os.path.exists(rgb_folder) or not os.path.exists(depth_folder) or not os.path.exists(timestamps_file):
        print("Error: One or more dataset paths are incorrect.")
        return

    tracker = TUMRGBDOpticalFlowTracker(rgb_folder, depth_folder, timestamps_file, model="farneback", visualization=False)

    flow_data = tracker.process()
    interactive_flow_viewer(flow_data, step=5)

    return flow_data