import cv2
import numpy as np
import os
import time
from utils import load_image_pair
from google.colab.patches import cv2_imshow  # Important: for image display in Colab


def feature_track(frame_associations,
                  p0,
                  gray_prev, bgr_prev, 
                  camera_matrix, 
                  dataset_path, 
                  feature_params, 
                  lk_params, 
                  dist_coeffs,
                  pnp_ransac_params, 
                  max_frames_to_process = 0, 
                  min_features_for_pnp = 5, 
                  min_features_redetect = 50, 
                  visualize_every_n_frames=10):


    # Initialize a list to store relative poses
    relative_poses = []
    
    # --- 4. Main Processing Loop (Steps 2 & 3) ---
    print("\n--- 4. Start Frame Processing Loop ---")
    start_time = time.time()

    # Determine the number of frames to process
    num_frames_to_process = len(frame_associations)
    print()
    if max_frames_to_process > 0:
        num_frames_to_process = min(len(frame_associations), max_frames_to_process + 1) # +1 because the loop starts from 1
        print(f"Note: Will only process the first {max_frames_to_process} frames (total {num_frames_to_process - 1} frame pairs).")

    # Loop through the frames starting from the second frame (index i=1)
    for i in range(1, num_frames_to_process):
        loop_start_time = time.time()
        print(f"\n--- Processing Frame {i}/{num_frames_to_process - 1} ---")

        # -- 4.1 Load the current frame --
        bgr_curr, depth_curr, gray_curr = None, None, None # Initialize to None
        load_success = False
        try:
            rgb_stamp_curr, rgb_file_curr, depth_stamp_curr, depth_file_curr = frame_associations[i]

            # !! Debug Print: Print the files being attempted to load !!
            print(f"Attempting to load RGB: {dataset_path}/{rgb_file_curr}")
            # print(f"Attempting to load Depth: {dataset_path}/{depth_file_curr}") # Optional

            bgr_curr, depth_curr = load_image_pair(rgb_file_curr, depth_file_curr, dataset_path)
            if bgr_curr is not None and depth_curr is not None:
                gray_curr = cv2.cvtColor(bgr_curr, cv2.COLOR_BGR2GRAY)
                load_success = True
            else:
                # load_image_pair already prints warnings internally
                pass

        except Exception as e:
            print(f"!! Error: Exception occurred while loading frame {i}: {e} !!")
            load_success = False # Ensure it's marked as failed

        # If loading failed, cannot process, skip this frame and force feature redetection in the next frame
        if not load_success:
            print(f"Skipping processing for frame {i}.")
            # gray_prev, depth_prev remain unchanged (state of i-1)
            p0 = None # Force feature redetection in the next iteration
            continue # Proceed to the next iteration

        # -- 4.2 Feature Tracking (Step 2) --
        good_new = None # Initialize successfully tracked points for this iteration

        # Check if there are valid feature points p0 from the previous iteration
        if p0 is None or len(p0) == 0:
            print(f"  Tracking: No feature points in the previous frame, attempting to redetect in the current frame ({i})...")
            p0 = cv2.goodFeaturesToTrack(gray_curr, mask = None, **feature_params)
            if p0 is None or len(p0) < min_features_for_pnp:
                print(f"  Tracking: Redetection failed or insufficient points, skipping.")
                gray_prev = gray_curr.copy() # Update state
                depth_prev = depth_curr.copy()
                p0 = None # Ensure redetection is attempted in the next iteration
                continue
            else:
                print(f"  Tracking: Redetected {len(p0)} points in the current frame ({i}). Cannot calculate relative pose.")
                # Cannot calculate pose, but update state for the next iteration
                gray_prev = gray_curr.copy()
                depth_prev = depth_curr.copy()
                # p0 is now the current frame's points, ready for tracking to i+1
                continue

        # Calculate optical flow
        # print(f"  Tracking: Tracking {len(p0)} points from frame {i-1} to frame {i}...")
        p1, st, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_curr, p0, None, **lk_params)

        # -- 4.3 Filter valid tracked points --
        R_relative = None # Reset relative pose results for the current frame
        t_relative = None
        factor = 1

        if p1 is not None and st is not None:
            st_flat = st.flatten()
            good_new = p1[st_flat == 1] # Successfully tracked points in the current frame
            good_old = p0[st_flat == 1] # Corresponding points in the previous frame
            print(f"  Tracking: Successfully tracked {len(good_new)} points.")

            # -- 4.4 Relative Pose Estimation (Step 3) --
            if len(good_new) >= min_features_for_pnp:
                # --- Get 3D points from the previous frame's depth map ---
                points_3d = []
                points_2d = []
                valid_indices_in_good_old = []
                for i_pt, pt_old in enumerate(good_old):
                    u, v = pt_old.ravel(); u_int, v_int = int(round(u)), int(round(v))
                    if 0 <= v_int < depth_prev.shape[0] and 0 <= u_int < depth_prev.shape[1]:
                        d = depth_prev[v_int, u_int]
                        if d > 0.1 and d < 10.0: # Valid depth check
                            z_3d = d / factor; x_3d = (u - cx) * z_3d / fx; y_3d = (v - cy) * z_3d / fy
                            points_3d.append([x_3d, y_3d, z_3d])
                            points_2d.append(good_new[i_pt])
                            valid_indices_in_good_old.append(i_pt)

                # --- Estimate pose using PnP RANSAC ---
                if len(points_3d) >= min_features_for_pnp:
                    points_3d = np.array(points_3d, dtype=np.float32)
                    points_2d = np.array(points_2d, dtype=np.float32)
                    # print(f"  Pose Estimation: Using {len(points_3d)} 3D-2D point pairs for PnP RANSAC...")
                    try:
                        success, rvec, tvec, inliers = cv2.solvePnPRansac(
                            points_3d, points_2d, camera_matrix, dist_coeffs, **pnp_ransac_params)
                        if success:
                        #     R_relative, _ = cv2.Rodrigues(rvec)
                        #     t_relative = tvec.flatten() # Ensure t is a 1D array
                        #     # R_cw, _ = cv2.Rodrigues(rvec)
                        #     # t_cw = tvec.flatten()

                        #     # R_relative = R_cw.T
                        #     # t_relative = -R_cw.T @ t_cw
                            rvec_refined, tvec_refined = cv2.solvePnPRefineLM(
                                points_3d, points_2d, camera_matrix, dist_coeffs, rvec, tvec)

                            # Convert the refined rotation vector to a rotation matrix
                            R_relative, _ = cv2.Rodrigues(rvec_refined)
                            # Flatten the refined translation vector to 1D
                            t_relative = tvec_refined.flatten()

                            # # Invert the transformation to obtain the relative pose from previous frame to current frame.
                            # # This step is crucial if the PnP returns the pose that maps object points
                            # # from the previous frame to the current camera's coordinate system.
                            # R_relative = R_cw.T
                            # t_relative = -R_cw.T @ t_cw
                            relative_poses.append({'R': R_relative, 't': t_relative, 'frame_index': i})
                            num_inliers = len(inliers) if inliers is not None else 0
                            print(f"  Pose Estimation: PnP successful! Inliers: {num_inliers}/{len(points_3d)}")
                            # Optional: Optimize good_new based on inliers (if needed)
                            # ...
                        else:
                            print(f"  Pose Estimation: PnP RANSAC failed to solve.")
                    except Exception as e:
                        print(f"  Pose Estimation: PnP RANSAC exception: {e}")
                else:
                    print(f"  Pose Estimation: Insufficient valid 3D points ({len(points_3d)}) for PnP.")
            else:
                print(f"  Pose Estimation: Insufficient successfully tracked points ({len(good_new)}) for PnP.")
        else:
            print(f"  Tracking: Optical flow calculation failed (p1 or st is None).")
            good_new = None # Mark tracking as failed

        # -- 4.5 Visualization (Colab, based on frequency) --
        if visualize_every_n_frames > 0 and i % visualize_every_n_frames == 0:
            print(f"--- Displaying tracking results for frame {i} ---")
            bgr_curr_display = bgr_curr.copy()
            temp_mask = np.zeros_like(bgr_curr)
            tracked_count_display = 0
            if p1 is not None and st is not None: # Ensure optical flow calculation has results
                tracked_count_display = len(good_new) if good_new is not None else 0
                # Use all successfully tracked points to display trajectory lines
                for j, (new, old) in enumerate(zip(p1[st_flat == 1], p0[st_flat == 1])):
                    a, b = new.ravel().astype(int)
                    c, d = old.ravel().astype(int)
                    cv2.line(temp_mask, (a, b), (c, d), (0, 255, 0), 1)
                    cv2.circle(bgr_curr_display, (a, b), 3, (0, 0, 255), -1) # Draw all successfully tracked points
            img_display = cv2.add(bgr_curr_display, temp_mask)
            info_text = f'Frame: {i}, Tracked: {tracked_count_display}'
            if R_relative is not None: info_text += f', Pose OK'
            cv2.putText(img_display, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
            cv2_imshow(img_display)
            # print(f"--- Display of frame {i} complete ---")

        # -- 4.6 Update State (Prepare for the next frame) --
        gray_prev = gray_curr.copy()
        # depth_curr_filtered = cv2.medianBlur(depth_curr, 5)
        # depth_prev = depth_curr_filtered.copy() # Use the filtered depth map as depth_prev for the next iteration
        depth_prev = depth_curr.copy() # Use the original depth map as depth_prev for the next iteration

        # Update the feature points p0 to track in the next iteration
        if good_new is not None and len(good_new) > 0:
            p0 = good_new.reshape(-1, 1, 2)
            # --- Feature Point Replenishment ---
            if len(p0) < min_features_redetect:
                # print(f"  Update: Low number of feature points ({len(p0)}), attempting to replenish...")
                redetection_mask = np.ones_like(gray_prev)
                for pt in p0:
                    x,y = pt.ravel().astype(int)
                    if 0 <= x < redetection_mask.shape[1] and 0 <= y < redetection_mask.shape[0]:
                        cv2.circle(redetection_mask, (x, y), 5, 0, -1)
                new_features = cv2.goodFeaturesToTrack(gray_prev, mask=redetection_mask, **feature_params)
                if new_features is not None:
                    # print(f"  Update: Replenished {len(new_features)} new feature points.")
                    if p0.size: p0 = np.concatenate((p0, new_features), axis=0)
                    else: p0 = new_features
                # else: print("  Update: Failed to replenish feature points.")
        else:
            # If optical flow failed or zero points tracked, redetection is necessary in the next iteration
            print(f"  Update: No valid tracked points, redetection will be performed in the next iteration.")
            p0 = None

        loop_end_time = time.time()
        print(f"--- Processing of frame {i} complete, time taken: {loop_end_time - loop_start_time:.3f} seconds ---")


    # --- 5. End ---
    end_time = time.time()
    print("\n--- 5. All frames processed ---")
    print(f"Successfully calculated {len(relative_poses)} relative poses.")
    total_time = end_time - start_time
    print(f"Total processing time: {total_time:.2f} seconds")
    if len(relative_poses) > 0:
        avg_time_per_pose = total_time / len(relative_poses)
        print(f"Average processing time per frame (with successful pose estimation): {avg_time_per_pose:.3f} seconds")

    return relative_poses