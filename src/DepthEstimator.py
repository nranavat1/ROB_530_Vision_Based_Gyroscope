import os
import numpy as np
import torch
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


class MidasDepthGenerator:
    def __init__(self, rgb_folder, output_folder, associated_file, model_type="DPT_Large"):
        self.rgb_folder = rgb_folder
        self.output_folder = output_folder
        self.associated_file = associated_file
        self.model_type = model_type

        os.makedirs(self.output_folder, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load Model
        self.model = torch.hub.load("intel-isl/MiDaS", model_type)
        self.model.to(self.device)
        self.model.eval()

        # Load transforms
        midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        if model_type in ["DPT_Large", "DPT_Hybrid"]:
            self.transform = midas_transforms.dpt_transform
        else:
            self.transform = midas_transforms.small_transform

    def load_association_list(self):
        rgb_depth_pairs = []
        with open(self.associated_file, 'r') as f:
            for line in f.readlines():
                if line.strip() and not line.startswith('#'):
                    values = line.strip().split()
                    if len(values) >= 4:
                        rgb_filename = os.path.basename(values[1])
                        depth_filename = os.path.basename(values[3])
                        rgb_depth_pairs.append((rgb_filename, depth_filename))
        return rgb_depth_pairs

    def run(self):
        rgb_depth_list = self.load_association_list()
        progress_bar = tqdm(total=len(rgb_depth_list), desc="Generating MiDaS Depth", unit="frame",leave=False)

        for rgb_filename, depth_filename in rgb_depth_list:
            rgb_path = os.path.join(self.rgb_folder, rgb_filename)
            output_path = os.path.join(self.output_folder, depth_filename.replace(".png", ".npy"))

            if not os.path.exists(rgb_path):
                print(f"Warning: {rgb_path} not found.")
                progress_bar.update(1)
                continue

            img = cv2.imread(rgb_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            sample = self.transform(img_rgb)
            if isinstance(sample, dict):
                input_tensor = sample["image"].to(self.device).unsqueeze(0)
            else:
                input_tensor = sample.to(self.device)

            with torch.no_grad():
                prediction = self.model(input_tensor)
                prediction = torch.nn.functional.interpolate(
                    prediction.unsqueeze(1),
                    size=img.shape[:2],
                    mode="bicubic",
                    align_corners=False,
                ).squeeze()

            # Invert Depth (In TUM, higher value means further)
            depth_map = -prediction.cpu().numpy()
            np.save(output_path, depth_map)

            # Save png and normalize
            depth_min = np.min(depth_map)
            depth_max = np.max(depth_map)
            depth_vis = 255 * (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
            depth_vis = depth_vis.astype(np.uint8)

            png_path = output_path.replace(".npy", ".png")
            cv2.imwrite(png_path, depth_vis)

            progress_bar.update(1)

        progress_bar.close()


class DepthEvaluator:
    def __init__(self, est_depth_folder, gt_depth_folder, associated_file):
        self.est_folder = est_depth_folder
        self.gt_folder = gt_depth_folder
        self.associated_file = associated_file
        self.pairs = self.load_associated_depth_pairs()

    def load_associated_depth_pairs(self):
        pairs = []
        with open(self.associated_file, 'r') as f:
            for line in f.readlines():
                if line.strip() and not line.startswith('#'):
                    values = line.strip().split()
                    if len(values) >= 4:
                        depth_filename = os.path.basename(values[3])
                        pairs.append(depth_filename)
        return pairs

    def scale_and_shift_align(self, pred, target, mask):
        A = np.stack([pred[mask], np.ones_like(pred[mask])], axis=1)
        B = target[mask]
        x, _, _, _ = np.linalg.lstsq(A, B, rcond=None)  # x = [scale, shift]
        return x[0] * pred + x[1]

    def evaluate(self, visualize_count=5):
        rmses = []
        maes = []
        rels = []

        print("\nEvaluating depth predictions with scale & shift alignment...")
        for i, filename in enumerate(tqdm(self.pairs)):
            est_path = os.path.join(self.est_folder, filename.replace(".png", ".npy"))
            gt_path = os.path.join(self.gt_folder, filename)

            if not os.path.exists(est_path) or not os.path.exists(gt_path):
                continue

            est_depth = np.load(est_path)
            gt_depth = cv2.imread(gt_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
            gt_depth = gt_depth / 5000.0 if np.max(gt_depth) > 255 else gt_depth

            if est_depth.shape != gt_depth.shape:
                est_depth = cv2.resize(est_depth, (gt_depth.shape[1], gt_depth.shape[0]), interpolation=cv2.INTER_CUBIC)

            valid_mask = (gt_depth > 0)
            est_depth_aligned = self.scale_and_shift_align(est_depth, gt_depth, valid_mask)

            rmse = np.sqrt(mean_squared_error(gt_depth[valid_mask], est_depth_aligned[valid_mask]))
            mae = mean_absolute_error(gt_depth[valid_mask], est_depth_aligned[valid_mask])
            rel = np.mean(np.abs(gt_depth[valid_mask] - est_depth_aligned[valid_mask]) / (gt_depth[valid_mask] + 1e-8))

            rmses.append(rmse)
            maes.append(mae)
            rels.append(rel)

            if i < visualize_count:
                def normalize_for_display(x):
                    x = np.nan_to_num(x)
                    x_min, x_max = np.percentile(x, 1), np.percentile(x, 99)
                    return np.clip((x - x_min) / (x_max - x_min + 1e-8), 0, 1)

                est_disp = normalize_for_display(est_depth_aligned)
                gt_disp = normalize_for_display(gt_depth)

                fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                axs[0].imshow(gt_disp, cmap='plasma')
                axs[0].set_title('Ground Truth')
                axs[1].imshow(est_disp, cmap='plasma')
                axs[1].set_title('Estimated (Aligned)')

                for ax in axs:
                    ax.axis('off')

                plt.suptitle(f"{filename} | RMSE={rmse:.3f} | REL={rel:.3f}")
                plt.show()

        print("\nEvaluation Results:")
        print(f"Average RMSE: {np.mean(rmses):.4f}")
        print(f"Average MAE: {np.mean(maes):.4f}")
        print(f"Average REL : {np.mean(rels):.4f}")

        return {
            'rmse': np.mean(rmses),
            'mae': np.mean(maes),
            'rel': np.mean(rels)
        }
