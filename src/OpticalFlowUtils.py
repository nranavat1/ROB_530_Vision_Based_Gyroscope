# --- Import Colab image display utility ---
from google.colab.patches import cv2_imshow  # Important: for image display in Colab
import cv2
from utils import associate_frames, load_image_pair
import numpy as np

def detectFeaturePoints(frame_associations,dataset_path, feature_params):
    # --- Initialization ---
    # (Assumes frame_associations has been obtained from Step 1)
    if not frame_associations:
        print("Error: Frame association list is empty.")
        exit()

    # Load the first frame
    try:
        rgb_stamp_prev, rgb_file_prev, depth_stamp_prev, depth_file_prev = frame_associations[0]
        bgr_prev, depth_prev = load_image_pair(rgb_file_prev, depth_file_prev, dataset_path)
        if bgr_prev is None or depth_prev is None:
            raise ValueError("Failed to load the first frame image")
    except Exception as e:
        print(f"Initialization error: {e}")
        exit()

    # Convert to grayscale
    gray_prev = cv2.cvtColor(bgr_prev, cv2.COLOR_BGR2GRAY)

    # Detect initial features (corners)
    p0 = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)

    if p0 is None:
        print("No initial features detected in the first frame.")
        exit()

    print(f"Detected {len(p0)} initial feature points.")
    return bgr_prev, gray_prev, p0


def createMask(bgr_prev):
    # Create a mask for drawing the trajectories (optional)
    mask = np.zeros_like(bgr_prev)


def opticalFlowTracking(frame_associations,bgr_prev, gray_prev, p0, dataset_path, feature_params, lk_params):

    # --- Main tracking loop ---
    # Note: In Colab, displaying too many images may flood output
    # You may choose to display only every Nth frame or test on fewer frames
    # for i in range(1, min(len(frame_associations), 50)):  # e.g. only process first 50
    for i in range(1, len(frame_associations)):
        try:
            rgb_stamp_curr, rgb_file_curr, depth_stamp_curr, depth_file_curr = frame_associations[i]
            bgr_curr, depth_curr = load_image_pair(rgb_file_curr, depth_file_curr, dataset_path)
            if bgr_curr is None or depth_curr is None:
                print(f"Warning: Skipping frame {i}, failed to load images.")
                continue
        except Exception as e:
            print(f"Error loading frame {i}: {e}")
            continue

        gray_curr = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow (Lucas-Kanade method)
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p0, None, **lk_params)

        # Filter out successfully tracked points
        if p1 is not None and st is not None:
            good_new = p1[st == 1]
            good_old = p0[st == 1]

            if len(good_new) < 5:
                print(f"Frame {i}: Too few tracked points ({len(good_new)}), skipping.")
                gray_prev = gray_curr.copy()
                p0 = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)
                if p0 is None:
                    print("Failed to re-detect features.")
                    break
                mask = createMask(bgr_prev)
                continue

            # Optionally keep tracking count print
            # print(f"Frame {i}: Successfully tracked {len(good_new)} points.")

        else:
            print(f"Frame {i}: Optical flow tracking failed (p1 is None or st is None).")
            gray_prev = gray_curr.copy()
            p0 = cv2.goodFeaturesToTrack(gray_prev, mask=None, **feature_params)
            if p0 is None:
                print("Failed to re-detect features.")
                break
            mask = createMask(bgr_prev)
            continue

        # --- (Here we would insert Step 3: Pose Estimation) ---
        # Inputs: good_old, good_new, depth_prev, camera_matrix
        # -------------------------------------------------------

        # --- Visualization (using cv2_imshow) ---
        VISUALIZE_EVERY_N_FRAMES = 10  # Display every 10 frames
        if i % VISUALIZE_EVERY_N_FRAMES == 0 or i == len(frame_associations) - 1:
            print(f"--- Displaying tracking result for frame {i} ---")
            bgr_curr_display = bgr_curr.copy()
            temp_mask = np.zeros_like(bgr_curr)

            for j, (new, old) in enumerate(zip(good_new, good_old)):
                a, b = new.ravel().astype(int)
                c, d = old.ravel().astype(int)
                temp_mask = cv2.line(temp_mask, (a, b), (c, d), (0, 255, 0), 1)
                bgr_curr_display = cv2.circle(bgr_curr_display, (a, b), 3, (0, 0, 255), -1)

            img_display = cv2.add(bgr_curr_display, temp_mask)
            cv2.putText(img_display, f'Frame: {i}, Tracked: {len(good_new)}',
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

            cv2_imshow(img_display)
            print(f"--- Frame {i} display complete ---")

        # --- End of Visualization ---

        # --- Update for next iteration ---
        gray_prev = gray_curr.copy()
        p0 = good_new.reshape(-1, 1, 2)

        # --- Feature point replenishment logic ---
        MIN_FEATURES_THRESHOLD = 50
        if len(p0) < MIN_FEATURES_THRESHOLD:
            # Optionally print
            # print(f"Frame {i}: Too few features ({len(p0)}), trying to replenish...")
            redetection_mask = np.ones_like(gray_prev)
            for pt in p0:
                x, y = pt.ravel().astype(int)
                cv2.circle(redetection_mask, (x, y), 5, 0, -1)
            new_features = cv2.goodFeaturesToTrack(gray_prev, mask=redetection_mask, **feature_params)
            if new_features is not None:
                # print(f"Frame {i}: Added {len(new_features)} new features.")
                if p0.size:
                    p0 = np.concatenate((p0, new_features), axis=0)
                else:
                    p0 = new_features
            # else:
                # print(f"Frame {i}: No new features found.")
            # mask = np.zeros_like(bgr_prev)  # Reset mask if not accumulating

    # --- No need for cv2.destroyAllWindows in Colab ---
    # cv2.destroyAllWindows()

    print("Feature tracking complete.")