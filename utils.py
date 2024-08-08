# utils.py 
import os
import shutil
import json
import numpy as np
import cv2
import open3d as o3d
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.transform import Rotation as R

import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def filter_image(image_path, h_squares, v_squares):
    try:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (h_squares, v_squares), None)
        return image_path, img, ret, corners
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return image_path, None, False, None

def save_image(image_path, img, valid_output_dir, invalid_output_dir, is_valid, corners, h_squares, v_squares):
    if is_valid:
        cv2.imwrite(os.path.join(valid_output_dir, os.path.basename(image_path)), img)
    else:
        cv2.drawChessboardCorners(img, (h_squares, v_squares), corners, is_valid)
        cv2.imwrite(os.path.join(invalid_output_dir, os.path.basename(image_path)), img)

def filter_and_save_images(image_paths, h_squares, v_squares, valid_output_dir, invalid_output_dir):
    if not os.path.exists(valid_output_dir):
        os.makedirs(valid_output_dir)
    if not os.path.exists(invalid_output_dir):
        os.makedirs(invalid_output_dir)

    valid_images = []
    invalid_images = []

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: filter_image(p, h_squares, v_squares), image_paths))

    for image_path, img, is_valid, corners in results:
        if is_valid:
            valid_images.append(image_path)
        else:
            invalid_images.append(image_path)
        save_image(image_path, img, valid_output_dir, invalid_output_dir, is_valid, corners, h_squares, v_squares)

    return valid_images, invalid_images

def upload_images(image_path):
    img = cv2.imread(image_path)
    return img

def detect_image_resolution(image_path):
    img = cv2.imread(image_path)
    if img is not None:
        height, width = img.shape[:2]
        return width, height
    else:
        raise ValueError(f"Error loading image {image_path}")

def approximate_intrinsic_parameters(width, height, fov_x=60):
    fx = width / (2 * np.tan(np.deg2rad(fov_x / 2)))
    fy = height / (2 * np.tan(np.deg2rad(fov_x / 2)))  # Assuming the same FOV for y direction
    cx = width / 2
    cy = height / 2
    mtx = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=np.float64)
    return mtx

def calibrate_camera(images, initial_mtx, correct_radial, correct_tangential, squares_x, squares_y, square_size):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((squares_y * squares_x, 3), np.float32)
    objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
    objp *= square_size  # Update object points to real world dimensions
    objpoints = []
    imgpoints = []

    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
    
    if objpoints and imgpoints:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        dist_coeffs = np.zeros((5, 1))  # Initialize distortion coefficients to zero
        
        if not correct_tangential[0]:
            flags |= cv2.CALIB_FIX_TANGENT_DIST
        if not correct_radial[0]:
            flags |= cv2.CALIB_FIX_K1
        if not correct_radial[1]:
            flags |= cv2.CALIB_FIX_K2
        if not correct_radial[2]:
            flags |= cv2.CALIB_FIX_K3

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], initial_mtx, dist_coeffs, flags=flags)
        return ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints
    else:
        raise ValueError("Chessboard corners not found in any image during camera calibration.")

def undistort_image(img, mtx, dist):
    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    return dst, newcameramtx

def process_images(images, correct_radial, correct_tangential, intrinsic_matrix, squares_x, squares_y, square_size):
    initial_images = images[:40]
    ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = calibrate_camera(initial_images, intrinsic_matrix, correct_radial, correct_tangential, squares_x, squares_y, square_size)
    
    undistorted_images = []
    for img in images:
        undistorted_img, new_mtx = undistort_image(img, mtx, dist)
        blurred_img = cv2.GaussianBlur(undistorted_img, (5, 5), 0)
        undistorted_images.append(blurred_img)
    return undistorted_images, mtx

def detect_chessboard_corners_3d(image_path, mtx, squares_x, squares_y, square_size):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image {image_path} during 3D chessboard detection")
        return None, None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pattern_size = (squares_x, squares_y)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objp = np.zeros((squares_y * squares_x, 3), np.float32)
        objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
        objp *= square_size  # Update object points to real world dimensions
        
        ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, None)
        if ret:
            projected_points, _ = cv2.projectPoints(objp, rvecs, tvecs, mtx, None)
            centroid_3d = np.mean(projected_points, axis=0)
            
            rot_mat, _ = cv2.Rodrigues(rvecs)
            chessboard_z = rot_mat[:, 2]
            camera_z = np.array([0, 0, 1])
            cos_angle = np.dot(chessboard_z, camera_z)
            angle_z = np.arccos(cos_angle)
            
            centroid_with_angle = np.append(centroid_3d.flatten(), angle_z)
            
            print(f"3D centroid and angle detected for {image_path}")
            return centroid_with_angle, np.array([np.linalg.norm(corners2[0] - corners2[1]), np.linalg.norm(corners2[0] - corners2[6])])
    
    return None, None

def detect_and_cluster(image_paths, intrinsic_matrix, squares_x, squares_y, square_size, num_clusters):
    features = []
    for img_path in image_paths:
        centroid, dimensions = detect_chessboard_corners_3d(img_path, intrinsic_matrix, squares_x, squares_y, square_size)
        if centroid is not None:
            features.append((img_path, np.hstack((centroid, dimensions))))
    
    paths, data = zip(*features) if features else ([], [])
    data = np.array(data)
    
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data[:, :4])  # Scaling the centroid data and the angle

    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    labels = kmeans.fit_predict(data_scaled)

    clusters = {label: [] for label in set(labels)}
    for idx, label in enumerate(labels):
        clusters[label].append(paths[idx])
    
    return clusters

def calculate_reprojection_error(objpoints, imgpoints, rvecs, tvecs, mtx, dist):
    total_error = 0
    squared_error_sum = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        squared_error_sum += error ** 2
    mean_error = total_error / len(objpoints)
    rmse = np.sqrt(squared_error_sum / len(objpoints))
    return mean_error, rmse

def calculate_reprojection_error_elite(objpoints, imgpoints, rvec, tvec, mtx, dist):
    total_error = 0
    squared_error_sum = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvec, tvec, mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error
        squared_error_sum += error ** 2
    mean_error = total_error / len(objpoints)
    rmse = np.sqrt(squared_error_sum / len(objpoints))
    return mean_error, rmse

def select_elite_images(features, intrinsic_matrix, squares_x, squares_y, square_size):
    elite_images = []
    for cluster_id, image_paths in features.items():
        images = [cv2.imread(img_path) for img_path in image_paths]
        ret, mtx, dist, rvecs, tvecs, objpoints, imgpoints = calibrate_camera(
            images, 
            intrinsic_matrix, 
            [True, True, True], 
            [True, True], 
            squares_x, 
            squares_y, 
            square_size
        )
        
        if not ret:
            continue
        
        reprojection_errors = {}
        for img_path in image_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
            if ret:
                objp = np.zeros((squares_y * squares_x, 3), np.float32)
                objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
                objp *= square_size  # Update object points to real world dimensions
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, mtx, dist)
                if ret:
                    mean_error, rmse = calculate_reprojection_error([objp], [corners2], [rvecs], [tvecs], mtx, dist)
                    reprojection_errors[img_path] = (mean_error, rmse)

        elite_image_path = min(reprojection_errors, key=lambda k: reprojection_errors[k][1])  # Using RMSE to select elite image
        elite_images.append(elite_image_path)

    return elite_images

import numpy as np
import cv2
import json
import os

def final_calibration(elite_images, intrinsic_matrix, squares_x, squares_y, square_size):
    objpoints = []
    imgpoints = []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Ensure the "elites" directory exists in the same directory as "undistorted_images"
    base_dir = os.path.dirname(os.path.dirname(elite_images[0]))
    elite_dir = os.path.join(base_dir, "elites")
    os.makedirs(elite_dir, exist_ok=True)

    for img_path in elite_images:
        img = cv2.imread(img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
        if ret:
            objp = np.zeros((squares_y * squares_x, 3), np.float32)
            objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
            objp *= square_size  # Update object points to real world dimensions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            objpoints.append(objp)
            imgpoints.append(corners2)

    if objpoints and imgpoints:
        flags = cv2.CALIB_USE_INTRINSIC_GUESS
        dist_coeffs = np.zeros((5, 1))  # Initialize distortion coefficients to zero
        
        # Assuming full correction for final calibration
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], intrinsic_matrix, dist_coeffs, flags=flags)
        
        for idx, img_path in enumerate(elite_images):
            rvec = np.asarray(rvecs[idx], dtype=np.float64).reshape(3, 1)
            tvec = np.asarray(tvecs[idx], dtype=np.float64).reshape(3, 1)
            
            # Compute rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            
            # Compute Euler angles (ZYX convention)
            sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
            singular = sy < 1e-6
            if not singular:
                x = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                x = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                y = np.arctan2(-rotation_matrix[2, 0], sy)
                z = 0
            
            euler_angles = [np.degrees(angle) for angle in (x, y, z)]
            
            # Transformation matrix (4x4)
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = tvec.flatten()
            
            # Prepare data for JSON
            data = {
                "translation_vector": tvec.flatten().tolist(),
                "rotation_vector": rvec.flatten().tolist(),
                "transformation_matrix": transformation_matrix.tolist(),
                "euler_angles": euler_angles
            }
            
            # Save JSON file in "elites" directory
            img_filename = os.path.basename(img_path)
            json_filename = os.path.splitext(img_filename)[0] + ".json"
            json_path = os.path.join(elite_dir, json_filename)
            with open(json_path, 'w') as json_file:
                json.dump(data, json_file, indent=4)
                
        return mtx, dist, rvecs, tvecs, objpoints, imgpoints
    else:
        raise ValueError("Chessboard corners not found in any elite image during final calibration.")

def calculate_mean_reprojection_error_all(image_paths, intrinsic_matrix, dist_coeffs, squares_x, squares_y, square_size):
    total_error = 0
    squared_error_sum = 0
    valid_image_count = 0

    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
        if ret:
            objp = np.zeros((squares_y * squares_x, 3), np.float32)
            objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
            objp *= square_size  # Update object points to real world dimensions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            ret, rvecs, tvecs = cv2.solvePnP(objp, corners2, intrinsic_matrix, dist_coeffs)
            if ret:
                mean_error, rmse = calculate_reprojection_error([objp], [corners2], [rvecs], [tvecs], intrinsic_matrix, dist_coeffs)
                total_error += mean_error
                squared_error_sum += rmse ** 2
                valid_image_count += 1

    if valid_image_count > 0:
        mean_reprojection_error = total_error / valid_image_count
        rmse = np.sqrt(squared_error_sum / valid_image_count)
    else:
        mean_reprojection_error = None
        rmse = None

    return mean_reprojection_error, rmse


import numpy as np
import cv2

def calculate_mean_reprojection_error_elites(image_paths, intrinsic_matrix, dist_coeffs, rvecs, tvecs, squares_x, squares_y, square_size):
    total_error = 0
    squared_error_sum = 0
    valid_image_count = 0

    print("rvecs", rvecs)
    print("dist_coeffs", dist_coeffs)

    for idx, img_path in enumerate(image_paths):
        img = cv2.imread(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (squares_x, squares_y), None)
        if ret:
            objp = np.zeros((squares_y * squares_x, 3), np.float32)
            objp[:, :2] = np.mgrid[0:squares_x, 0:squares_y].T.reshape(-1, 2)
            objp *= square_size  # Update object points to real world dimensions


            print("image est", img_path)
            print("objp [0]", objp[ 0])
            print("objp [7]", objp[ 6])
            print("objp[-7]",objp[len(objp)-7])
            print("objp [last]", objp[len(objp)-1])


            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            print("image est 1", img_path)
            print("corners2[0]", corners2[0])
            print("corners2[7]", corners2[6])
            print("corners2[-7]", corners2[len(corners2) - 7])
            print("corners2[last]", corners2[len(corners2) - 1])

            print("rvecs[idx]", rvecs[idx])
            print("tvecs[idx]", tvecs[idx])
            print("rvecs", rvecs)
            rvec = np.asarray(rvecs[idx], dtype=np.float32).reshape(3, 1)
            tvec = np.asarray(tvecs[idx], dtype=np.float32).reshape(3, 1)
            objpoints = [objp]
            imgpoints = [corners2]

            mean_error_image = 0
            squared_error_sum_image = 0
            for j in range(len(objpoints)):
                imgpoints2, _ = cv2.projectPoints(objpoints[j], rvec, tvec, intrinsic_matrix, dist_coeffs)
                
                error = cv2.norm(imgpoints[j], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
                mean_error_image += error
                squared_error_sum_image += error ** 2
            mean_error_image /= len(objpoints)
            rmse_image = np.sqrt(squared_error_sum_image / len(objpoints))

            total_error += mean_error_image
            squared_error_sum += rmse_image ** 2
            valid_image_count += 1
        else:
            print(f"Chessboard corners not found in {img_path}")

    if valid_image_count > 0:
        mean_reprojection_error = total_error / valid_image_count
        rmse = np.sqrt(squared_error_sum / valid_image_count)
    else:
        mean_reprojection_error = None
        rmse = None

    return mean_reprojection_error, rmse


import os

def match_point_clouds_with_images(image_paths, point_cloud_paths, max_time_diff=0.03):
    matched_pairs = []
    for img_path in image_paths:
        try:
            # Extracting timestamp from the image filename
            img_time = int(os.path.splitext(os.path.basename(img_path))[0])
            print(f"Image path: {img_path}, Extracted timestamp: {img_time}")

            # Finding the closest point cloud timestamp
            closest_pc_path = min(point_cloud_paths, key=lambda pc: abs(int(os.path.splitext(os.path.basename(pc))[0]) - img_time))
            closest_pc_time = int(os.path.splitext(os.path.basename(closest_pc_path))[0])
            print(f"Closest point cloud path: {closest_pc_path}, Extracted timestamp: {closest_pc_time}")

            # Calculating the time difference in nanoseconds
            time_diff = abs(closest_pc_time - img_time)
            # Converting the time difference to seconds
            time_diff_seconds = time_diff / 1_000_000_000.0
            print(f"Time Difference: {time_diff_seconds} seconds")

            if time_diff_seconds <= max_time_diff:
                matched_pairs.append((img_path, closest_pc_path))
                print(f"Matched: {img_path} with {closest_pc_path}")
            else:
                print(f"Skipped: {img_path} with {closest_pc_path} due to large time difference")

        except ValueError as e:
            print(f"Error parsing timestamp for {img_path} or {closest_pc_path}: {e}")
    return matched_pairs

"""import os

def match_point_clouds_with_images(image_paths, point_cloud_paths, max_time_diff=0.03):
    matched_pairs = []
    for img_path in image_paths:
        try:
            # Extracting timestamp from the image filename
            img_time = int(os.path.splitext(os.path.basename(img_path))[0])
            print(f"Image path: {img_path}, Extracted timestamp: {img_time}")

            # Filter point clouds that are after the image timestamp
            later_point_clouds = [pc for pc in point_cloud_paths if int(os.path.splitext(os.path.basename(pc))[0]) > img_time]
            
            if len(later_point_clouds) < 2:
                print(f"Not enough point clouds after the image timestamp for: {img_path}")
                continue
            
            # Find the second closest point cloud timestamp that is later than the image timestamp
            sorted_pc_paths = sorted(later_point_clouds, key=lambda pc: int(os.path.splitext(os.path.basename(pc))[0]))
            closest_pc_path = sorted_pc_paths[1]
            closest_pc_time = int(os.path.splitext(os.path.basename(closest_pc_path))[0])
            print(f"Second closest point cloud path: {closest_pc_path}, Extracted timestamp: {closest_pc_time}")

            # Calculating the time difference in nanoseconds
            time_diff = closest_pc_time - img_time
            # Converting the time difference to seconds
            time_diff_seconds = time_diff / 1_000_000_000.0
            print(f"Time Difference: {time_diff_seconds} seconds")

            if time_diff_seconds <= max_time_diff:
                matched_pairs.append((img_path, closest_pc_path))
                print(f"Matched: {img_path} with {closest_pc_path}")
            else:
                print(f"Skipped: {img_path} with {closest_pc_path} due to large time difference")

        except ValueError as e:
            print(f"Error parsing timestamp for {img_path} or {closest_pc_path}: {e}")
    return matched_pairs
"""
"""
def save_pairs(matched_pairs, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for i, (img_path, pc_path) in enumerate(matched_pairs):
        pair_dir = os.path.join(output_dir, f"pair_{i}")
        os.makedirs(pair_dir)
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            continue
        if not os.path.exists(pc_path):
            print(f"Point cloud file not found: {pc_path}")
            continue
        shutil.copy(img_path, os.path.join(pair_dir, os.path.basename(img_path)))
        shutil.copy(pc_path, os.path.join(pair_dir, os.path.basename(pc_path)))

    with open(os.path.join(output_dir, "matched_pairs.json"), "w") as f:
        json.dump(matched_pairs, f)
"""
import os
import shutil
import json

def save_pairs(matched_pairs, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    
    for i, (img_path, pc_path) in enumerate(matched_pairs):
        pair_dir = os.path.join(output_dir, f"pair_{i}")
        os.makedirs(pair_dir)
        if not os.path.exists(img_path):
            print(f"Image file not found: {img_path}")
            continue
        if not os.path.exists(pc_path):
            print(f"Point cloud file not found: {pc_path}")
            continue
        
        # Copy image file
        shutil.copy(img_path, os.path.join(pair_dir, os.path.basename(img_path)))
        
        # Copy point cloud file
        shutil.copy(pc_path, os.path.join(pair_dir, os.path.basename(pc_path)))

        # Determine the corresponding JSON file path
        base_dir = os.path.dirname(os.path.dirname(img_path))
        elite_dir = os.path.join(base_dir, "elites")
        json_filename = os.path.splitext(os.path.basename(img_path))[0] + ".json"
        json_path = os.path.join(elite_dir, json_filename)

        # Copy and rename JSON file if it exists
        if os.path.exists(json_path):
            target_json_path = os.path.join(pair_dir, "T_Chessboard_in_Camera.json")
            shutil.copy(json_path, target_json_path)
        else:
            print(f"JSON file not found: {json_path}")

    # Save matched pairs as a JSON file
    with open(os.path.join(output_dir, "matched_pairs.json"), "w") as f:
        json.dump(matched_pairs, f)

import os
import shutil
import numpy as np
import open3d as o3d

def create_and_label_bounding_box(pcd, picked_indices):
    if len(picked_indices) < 2:
        raise ValueError("Please pick at least two points to define the bounding box.")

    selected_points = np.asarray(pcd.points)[picked_indices]
    min_bound = selected_points.min(axis=0)
    max_bound = selected_points.max(axis=0)
    points = np.asarray(pcd.points)
    inside_box = np.all((points >= min_bound) & (points <= max_bound), axis=1)

    return inside_box

def load_point_cloud(file_path):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} not found.")
        point_cloud = np.loadtxt(file_path, delimiter=',', skiprows=1)
        return point_cloud
    except Exception as e:
        print(f"Error loading point cloud {file_path}: {e}")
        return None

class CustomVisualizer:
    def __init__(self):
        self.pcd = None
        self.selected_indices = []

    def load_point_cloud(self, file_path):
        print(f"Loading point cloud from: {file_path}")
        point_cloud_data = load_point_cloud(file_path)
        if point_cloud_data is None or point_cloud_data.size == 0:
            raise ValueError(f"Failed to load point cloud from: {file_path}")
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])

    def run_visualizer(self, pair_dir, pc_file):
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(self.pcd)
        vis.run()
        vis.destroy_window()

        self.selected_indices = vis.get_picked_points()
        self.annotate_and_save_point_cloud(pair_dir, pc_file)
        self.visualize_labeled_point_cloud(pair_dir, pc_file)

    def annotate_and_save_point_cloud(self, pair_dir, pc_file):
        if self.selected_indices:
            inside_box = create_and_label_bounding_box(self.pcd, self.selected_indices)
            point_cloud_data = np.asarray(self.pcd.points)
            labeled_point_cloud = np.hstack((point_cloud_data, np.zeros((point_cloud_data.shape[0], 1))))
            labeled_point_cloud[inside_box, -1] = 1
        else:
            labeled_point_cloud = np.hstack((np.asarray(self.pcd.points), np.zeros((np.asarray(self.pcd.points).shape[0], 1))))

        labeled_point_cloud_path = os.path.join(pair_dir, f"annotated_{pc_file}")
        np.savetxt(labeled_point_cloud_path, labeled_point_cloud, delimiter=" ")
        print(f"Annotated point cloud saved at {labeled_point_cloud_path}")

    def visualize_labeled_point_cloud(self, pair_dir, pc_file):
        labeled_file_path = os.path.join(pair_dir, f"annotated_{pc_file}")
        point_cloud_data = np.loadtxt(labeled_file_path, delimiter=" ")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
        colors = np.zeros((point_cloud_data.shape[0], 3))
        colors[point_cloud_data[:, 3] == 0] = [0, 0, 1]  # Blue for label 0
        colors[point_cloud_data[:, 3] == 1] = [1, 0, 0]  # Red for label 1
        pcd.colors = o3d.utility.Vector3dVector(colors)

        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

def get_next_point_cloud(pair_dirs, current_index):
    next_index = current_index + 1
    if next_index < len(pair_dirs):
        return pair_dirs[next_index], next_index
    return None, None

import os
import shutil
import json
import numpy as np
import open3d as o3d
from scipy.spatial import ConvexHull

import numpy as np
import open3d as o3d

#def create_text_3d(text, pos, direction=None, degree=0.0, density=10, font_size=1):
    #"""
    #Create a 3D text object using lines to simulate text.
    #"""
    #text_3d = o3d.geometry.TriangleMesh.create_coordinate_frame(size=font_size, origin=pos)
    #return text_3d

def visualize_with_rectangle_and_bases(point_cloud_data, corners_3d, O, X, Y, Z):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud_data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(np.array([[0, 0, 1]] * len(pcd.points)))  # All points in blue

    # Create the lines for the rectangle and coordinate systems
    lines = [
        [0, 1], [1, 2], [2, 3], [3, 0],  # Rectangle edges in black
        [4, 5], [4, 6], [4, 7],          # Coordinate system axes in red, green, blue
        [8, 9], [8, 10], [8, 11]         # Original coordinate system axes in red, green, blue
    ]

    # Points for the lines
    line_points = np.vstack([corners_3d, O, O + X, O + Y, O + Z, [0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    line_colors = [[0, 0, 0] for _ in range(4)] + [[1, 0, 0], [0, 1, 0], [0, 0, 1]] + [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(line_colors)

    # Create spheres for the corners and add labels
    labels = ['A', 'B', 'C', 'D']
    label_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]]  # Red, Green, Blue, Yellow
    spheres = []
    text_labels = []

    
    for i, corner in enumerate(corners_3d):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.05)
        sphere.translate(corner)
        sphere.paint_uniform_color(label_colors[i])
        spheres.append(sphere)

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.add_geometry(line_set)
    for sphere in spheres:
        vis.add_geometry(sphere)

    vis.run()
    vis.destroy_window()

def extract_labeled_points(pair_dir):
    labeled_point_files = [f for f in os.listdir(pair_dir) if f.startswith("annotated_") and f.endswith(".txt")]
    labeled_points = []

    for file in labeled_point_files:
        file_path = os.path.join(pair_dir, file)
        data = np.loadtxt(file_path, delimiter=' ')  # Adjust delimiter if necessary
        points_with_labels = data[data[:, 3] == 1]
        labeled_points.append(points_with_labels[:, :3])  # Extract XYZ coordinates of labeled points

    return np.vstack(labeled_points) if labeled_points else np.array([])


def detect_plane_ransac(points, distance_threshold=0.01, ransac_n=3, num_iterations=1000):
    if points.size == 0:
        raise ValueError("No labeled points found.")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    plane_points = points[inliers]
    return plane_model, plane_points


def project_points_to_plane(plane_model, points):
    # Calculate plane normal and point on the plane
    plane_normal = plane_model[:3]
    plane_point = -plane_model[3] * plane_normal / np.dot(plane_normal, plane_normal)
    
    # Define a local coordinate system on the plane
    z_axis = plane_normal / np.linalg.norm(plane_normal)
    if np.abs(z_axis[0]) > np.abs(z_axis[1]):
        x_axis = np.array([-z_axis[2], 0, z_axis[0]])
    else:
        x_axis = np.array([0, -z_axis[2], z_axis[1]])
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    
    # Project points onto the plane
    points_rel = points - plane_point
    points_2d = np.column_stack((np.dot(points_rel, x_axis), np.dot(points_rel, y_axis)))
    
    return points_2d, (x_axis, y_axis, plane_point)

def minimum_bounding_rectangle(hull_points):
    pi2 = np.pi / 2.

    # Calculate edge angles
    edges = np.diff(hull_points, axis=0)
    edge_angles = np.arctan2(edges[:, 1], edges[:, 0])

    # Normalize angles to range between 0 and pi/2
    edge_angles = np.abs(np.mod(edge_angles, pi2))
    edge_angles = np.unique(edge_angles)

    # Initialize minimum area as infinity
    min_area = float('inf')
    best_bbox = None

    for angle in edge_angles:
        # Create rotation matrix for the current angle
        R = np.array([
            [np.cos(angle), np.cos(angle - pi2)],
            [np.sin(angle), np.sin(angle - pi2)]
        ])

        # Rotate the hull points
        rot_points = np.dot(hull_points, R)

        # Find the bounding box of the rotated points
        min_x, min_y = np.min(rot_points, axis=0)
        max_x, max_y = np.max(rot_points, axis=0)
        area = (max_x - min_x) * (max_y - min_y)

        if area < min_area:
            min_area = area
            best_bbox = np.array([
                [min_x, min_y],
                [max_x, min_y],
                [max_x, max_y],
                [min_x, max_y]
            ]).dot(R.T)

    # Reorder the corners in counter-clockwise order
    if best_bbox is not None:
        best_bbox = reorder_corners_ccw(best_bbox)

    print("Best Bounding Box:", best_bbox)
    print("A:", best_bbox[0])
    print("B:", best_bbox[1])
    print("C:", best_bbox[2])
    print("D:", best_bbox[3])
    rearranged_bbox =[best_bbox[2], best_bbox[1], best_bbox[0], best_bbox[3]]

    return rearranged_bbox

def reorder_corners_ccw(corners):
    # Ensure corners are ordered as: bottom-left, bottom-right, top-right, top-left
    # Compute centroid
    centroid = np.mean(corners, axis=0)
    
    # Calculate angles of corners relative to centroid
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    
    # Sort corners by angle in counter-clockwise order
    sort_indices = np.argsort(angles)
    sorted_corners = corners[sort_indices]
    
    # Identify each corner based on sorted positions
    bottom_left = sorted_corners[0]
    top_left = sorted_corners[1]
    top_right = sorted_corners[2]
    bottom_right = sorted_corners[3]

    return np.array([bottom_left, bottom_right, top_right, top_left])

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def process_pair(pair_dir, pc_file, intrinsic_matrix, dist_coeffs, squares_x, squares_y, square_size, distance_vertical_edge, distance_horizontal_edge):
        
    custom_visualizer = CustomVisualizer()
    print(f"Processing pair in directory: {pair_dir}")
    print(f"Point cloud file: {pc_file}")
    
    pc_file_path = os.path.join(pair_dir, pc_file)
    custom_visualizer.load_point_cloud(pc_file_path)

    pair_index = os.path.basename(pair_dir).split('_')[-1]
    corners_camera_file = os.path.join(pair_dir, f"corners_in_camera_{pair_index}.txt")

    # Check if corners in camera already detected and saved
    if os.path.exists(corners_camera_file):
        print(f"Corners in camera already detected and saved in {corners_camera_file}")
        return

    while True:
        custom_visualizer.run_visualizer(pair_dir, pc_file)

        labeled_points = extract_labeled_points(pair_dir)
        if labeled_points.size == 0:
            print("No labeled points found.")
            return

        plane_model, plane_points = detect_plane_ransac(labeled_points)
        print("Plane model:", plane_model)

        points_2d, (x_axis, y_axis, plane_point) = project_points_to_plane(plane_model, plane_points)
        hull = ConvexHull(points_2d)
        hull_points = points_2d[hull.vertices]
        bbox = minimum_bounding_rectangle(hull_points)
        print("Optimal Rectangle Corners:", bbox)

        corners_3d = np.dot(bbox, np.vstack([x_axis, y_axis])) + plane_point
        print("3D Rectangle Corners:", corners_3d)

        if len(corners_3d) != 4:
            print("Error: Incorrect number of 3D corners detected.")
            return

        A_3d, B_3d, C_3d, D_3d = corners_3d

        O, X, Y, Z = calculate_base_relative_to_rectangle_LIDAR(A_3d, B_3d, C_3d, D_3d)

        print("Origin O:", O)
        print("X vector:", X)
        print("Y vector:", Y)
        print("Z vector:", Z)

        R = np.vstack([X, Y, Z]).T
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = O

        print("Transformation Matrix:")
        print(T)

        corners_file_path = os.path.join(pair_dir, f"corners_in_lidar_{pair_index}.txt")
        labels = ["A", "B", "C", "D"]
        with open(corners_file_path, "w") as f:
            for label, corner in zip(labels, corners_3d):
                f.write(f"{label} {corner[0]} {corner[1]} {corner[2]}\n")
        print(f"3D Rectangle Corners saved to {corners_file_path}")

        transformation_file_path = os.path.join(pair_dir, "T_Lidar_B.json")
        with open(transformation_file_path, "w") as f:
            json.dump(T.tolist(), f)
        print(f"Transformation matrix saved to {transformation_file_path}")

        T_B_Lidar = calculate_inverse_transformation(T)
        print("Inverse Transformation Matrix:")
        print(T_B_Lidar)

        inverse_transformation_file_path = os.path.join(pair_dir, "T_B_Lidar.json")
        save_transformation(T_B_Lidar, inverse_transformation_file_path)
        print(f"Inverse transformation (Target-LiDAR) matrix saved to {inverse_transformation_file_path}")

        point_cloud_data = load_point_cloud(os.path.join(pair_dir, pc_file))
        visualize_with_rectangle_and_bases(point_cloud_data, corners_3d, O, X, Y, Z)

        user_input = input("Press 'Y' to save the detection of the target, 'N' to retry, or 'D' to delete and move to the next pair: ").strip().lower()
        if user_input == 'y':
            break
        elif user_input == 'n':
            annotated_file_path = os.path.join(pair_dir, f"annotated_{pc_file}")
            if os.path.exists(annotated_file_path):
                os.remove(annotated_file_path)
        elif user_input == 'd':
            shutil.rmtree(pair_dir)
            return  # Exit the function if the pair is deleted
        
    image_files = [f for f in os.listdir(pair_dir) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
    for image_file in image_files:
        image_path = os.path.join(pair_dir, image_file)

        camera_target_matrix = compute_camera_target_transformation(
            image_path, distance_vertical_edge, distance_horizontal_edge
        )
        display_camera_target_visualization(image_path, intrinsic_matrix, dist_coeffs, camera_target_matrix, squares_x, squares_y, square_size, distance_vertical_edge, distance_horizontal_edge)

    T_Camera_Lidar = calculate_camera_lidar_transformation(camera_target_matrix, T_B_Lidar)

    euler_angles, translation_distances = extract_euler_angles_and_translation(T_Camera_Lidar)

    save_camera_lidar_transformation(pair_dir, T_Camera_Lidar, euler_angles, translation_distances)

    

import numpy as np
import cv2
import matplotlib.pyplot as plt
import streamlit as st

def display_camera_target_visualization(image_path, intrinsic_matrix, dist_coeffs, camera_target_matrix, number_vertical_squares, number_horizontal_squares, square_size, distance_vertical_edge, distance_horizontal_edge):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path} during visualization")

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    # Extract rotation and translation from the camera_target_matrix
    rotation_matrix = camera_target_matrix[:3, :3]
    translation_vector = camera_target_matrix[:3, 3]

    nv = number_vertical_squares - 1
    nh = number_horizontal_squares - 1
    epsilon = square_size
    dv = distance_vertical_edge
    dh = distance_horizontal_edge

    # Define the base points in the target frame (origin and axes)
    origin = np.array([0, 0, 0], dtype=np.float32)
    x_axis = np.array([1, 0, 0], dtype=np.float32)  # 1 m along the X-axis
    y_axis = np.array([0, 1, 0], dtype=np.float32)  # 1 m along the Y-axis
    z_axis = np.array([0, 0, 1], dtype=np.float32)  # 1 m along the Z-axis
    A_3d = np.array([0, 0, 0], dtype=np.float32)
    B_3d = np.array([nv * epsilon + 2 * dv, 0, 0], dtype=np.float32)
    C_3d = np.array([nv * epsilon + 2 * dv, nh * epsilon + 2 * dh, 0], dtype=np.float32)
    D_3d = np.array([0, nh * epsilon + 2 * dh, 0], dtype=np.float32)

    o_cam = np.array([0, 0, 0], dtype=np.float32)
    x_axis_cam = np.array([0.1, 0, 0], dtype=np.float32)
    y_axis_cam = np.array([0, 0.1, 0], dtype=np.float32)
    z_axis_cam = np.array([0, 0, 0.1], dtype=np.float32)

    points_cam_3d = np.array([o_cam, x_axis_cam, y_axis_cam, z_axis_cam], dtype=np.float32)

    # Debugging prints
    print(f"points_cam_3d: {points_cam_3d}")
    print(f"intrinsic_matrix: {intrinsic_matrix}")
    print(f"dist_coeffs: {dist_coeffs}")

    # Ensure points are non-empty and of correct type
    if points_cam_3d.size == 0:
        raise ValueError("Error: points_cam_3d is empty.")
    if not (points_cam_3d.dtype == np.float32 or points_cam_3d.dtype == np.float64):
        raise TypeError("Error: points_cam_3d must be of type float32 or float64.")

    points_cam_2d, _ = cv2.projectPoints(points_cam_3d, np.zeros(3), np.zeros(3), intrinsic_matrix, dist_coeffs)
    points_cam_2d = points_cam_2d.reshape(-1, 2).astype(int)

    # Transform the base points to the camera frame
    points_3d = np.array([origin, x_axis, y_axis, z_axis, A_3d, B_3d, C_3d, D_3d], dtype=np.float32)
    points_3d_cam = (rotation_matrix @ points_3d.T).T + translation_vector

    # Debugging prints
    print(f"rotation_matrix: {rotation_matrix}")
    print(f"translation_vector: {translation_vector}")
    print(f"points_3d_cam: {points_3d_cam}")

    # Project the 3D points to 2D image plane
    points_2d, _ = cv2.projectPoints(points_3d_cam, np.zeros(3), np.zeros(3), intrinsic_matrix, dist_coeffs)
    points_2d = points_2d.reshape(-1, 2).astype(int)

    # Debugging prints
    print(f"points_2d: {points_2d}")

    # Save the 3D coordinates of the points A, B, C, and D
    pair_index = os.path.basename(os.path.dirname(image_path)).split('_')[-1]
    corners_file_path = os.path.join(os.path.dirname(image_path), f"corners_in_camera_{pair_index}.txt")
    points_3d_cam_flat = points_3d_cam[4:]  # A, B, C, D points

    with open(corners_file_path, 'w') as f:
        for label, point in zip(['A', 'B', 'C', 'D'], points_3d_cam_flat):
            f.write(f"{label} {point[0]} {point[1]} {point[2]}\n")

    # Plot the origin and axes
    origin_2d = points_2d[0]
    x_axis_2d = points_2d[1]
    y_axis_2d = points_2d[2]
    z_axis_2d = points_2d[3]
    a_2d = points_2d[4]
    b_2d = points_2d[5]
    c_2d = points_2d[6]
    d_2d = points_2d[7]
    origin_cam_2d = points_cam_2d[0]
    x_axis_cam_2d = points_cam_2d[1]
    y_axis_cam_2d = points_cam_2d[2]
    z_axis_cam_2d = points_cam_2d[3]

    ax.scatter(0, 0, c='green', s=100, edgecolors='black')

    ax.scatter(origin_2d[0], origin_2d[1], c='yellow', s=100, edgecolors='black')
    ax.scatter(a_2d[0], a_2d[1], c='yellow', s=100, edgecolors='black')
    ax.scatter(b_2d[0], b_2d[1], c='yellow', s=100, edgecolors='black')
    ax.scatter(c_2d[0], c_2d[1], c='yellow', s=100, edgecolors='black')
    ax.scatter(d_2d[0], d_2d[1], c='yellow', s=100, edgecolors='black')
    ax.scatter(origin_cam_2d[0], origin_cam_2d[1], c='black', s=100, edgecolors='black')
    ax.scatter(x_axis_cam_2d[0], x_axis_cam_2d[1], c='red', s=100, edgecolors='black')
    ax.scatter(y_axis_cam_2d[0], y_axis_cam_2d[1], c='green', s=100, edgecolors='black')
    ax.scatter(z_axis_cam_2d[0], z_axis_cam_2d[1], c='blue', s=100, edgecolors='black')

    ax.plot([origin_2d[0], x_axis_2d[0]], [origin_2d[1], x_axis_2d[1]], 'r-', linewidth=2)
    ax.plot([origin_2d[0], y_axis_2d[0]], [origin_2d[1], y_axis_2d[1]], 'g-', linewidth=2)
    ax.plot([origin_2d[0], z_axis_2d[0]], [origin_2d[1], z_axis_2d[1]], 'b-', linewidth=2)

    ax.plot([origin_cam_2d[0], x_axis_cam_2d[0]], [origin_2d[1], x_axis_cam_2d[1]], 'r-', linewidth=2)
    ax.plot([origin_cam_2d[0], y_axis_cam_2d[0]], [origin_2d[1], y_axis_cam_2d[1]], 'g-', linewidth=2)
    ax.plot([origin_cam_2d[0], z_axis_cam_2d[0]], [origin_2d[1], z_axis_cam_2d[1]], 'b-', linewidth=2)

    ax.text(x_axis_2d[0], x_axis_2d[1], 'X', color='red', fontsize=12)
    ax.text(y_axis_2d[0], y_axis_2d[1], 'Y', color='green', fontsize=12)
    ax.text(z_axis_2d[0], z_axis_2d[1], 'Z', color='blue', fontsize=12)

    ax.text(x_axis_cam_2d[0], x_axis_cam_2d[1], 'X', color='red', fontsize=12)
    ax.text(y_axis_cam_2d[0], y_axis_cam_2d[1], 'Y', color='green', fontsize=12)
    ax.text(z_axis_cam_2d[0], z_axis_cam_2d[1], 'Z', color='blue', fontsize=12)

    ax.text(a_2d[0], a_2d[1], 'A', color='black', fontsize=12)
    ax.text(b_2d[0], b_2d[1], 'B', color='black', fontsize=12)
    ax.text(c_2d[0], c_2d[1], 'C', color='black', fontsize=12)
    ax.text(d_2d[0], d_2d[1], 'D', color='black', fontsize=12)

    st.pyplot(fig)


def visualize_and_label_point_clouds(intrinsic_matrix, dist_coeffs, squares_x, squares_y, square_size, distance_vertical_edge, distance_horizontal_edge):
    base_dir = "lidar-camera_pairs"
    if not os.path.exists(base_dir):
        raise ValueError(f"{base_dir} directory does not exist.")

    pair_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

    current_index = 0
    while current_index < len(pair_dirs):
        pair_dir = pair_dirs[current_index]
        point_cloud_files = [f for f in os.listdir(pair_dir) if f.endswith(".txt") and not f.startswith("annotated_")]

        # Display the single image for this pair
        image_files = [f for f in os.listdir(pair_dir) if f.endswith(".png") or f.endswith(".jpg") or f.endswith(".jpeg")]
        if image_files:
            image_path = os.path.join(pair_dir, image_files[0])
            st.image(image_path, caption=f"Image for {pair_dir}", use_column_width=True)
        
        for pc_file in point_cloud_files:
            try:
                process_pair(pair_dir, pc_file, intrinsic_matrix, dist_coeffs, squares_x, squares_y, square_size, distance_vertical_edge, distance_horizontal_edge)
            except FileNotFoundError:
                print(f"Directory {pair_dir} has been deleted, skipping to the next one.")
                continue

        current_index += 1
  
def reorder_corners(corners):
    # Ensure corners are ordered as: down-right, up-right, up-left, down-left
    centroid = np.mean(corners, axis=0)
    angles = np.arctan2(corners[:, 1] - centroid[1], corners[:, 0] - centroid[0])
    sort_indices = np.argsort(angles)
    sorted_corners = corners[sort_indices]

    # Identify each corner based on sorted positions
    down_right = sorted_corners[0]
    up_right = sorted_corners[1]
    up_left = sorted_corners[2]
    down_left = sorted_corners[3]

    # Ensure the corners are in the correct order (down-right, up-right, up-left, down-left)
    if np.sign(np.cross(up_right - down_right, down_left - down_right)[2]) < 0:
        down_left, up_left = up_left, down_left

    return np.array([down_right, up_right, up_left, down_left])

def calculate_base_relative_to_rectangle_LIDAR(A_3d, B_3d, C_3d, D_3d):
        
    # Define the origin and axes
    down_right = B_3d
    up_right = A_3d
    up_left = D_3d
    down_left = C_3d
    
    
    X = (B_3d-A_3d) / np.linalg.norm(B_3d-A_3d)
    
    Y = (D_3d - A_3d) / np.linalg.norm(D_3d - A_3d)
    
    # Z-axis: according to right-hand rule
    Z = np.cross(X, Y)
    
    return A_3d, X, Y, Z

def calculate_base_relative_to_rectangle_camera(A_3d, B_3d, C_3d, D_3d):
    """
    Calculate the coordinate system base relative to the rectangle in the camera image.

    Parameters:
    A_3d, B_3d, C_3d, D_3d: np.array
        3D coordinates of the corners A, B, C, D of the rectangle.

    Returns:
    O, X, Y, Z: np.array
        Origin and basis vectors of the coordinate system.
    """
    # Define the origin at B
    O = B_3d

    # X-axis from B to A
    X = (A_3d - B_3d) / np.linalg.norm(A_3d - B_3d)

    # Y-axis from B to C
    Y = (C_3d - B_3d) / np.linalg.norm(C_3d - B_3d)

    # Z-axis using right-hand rule
    Z = np.cross(X, Y)

    return O, X, Y, Z

import numpy as np
import cv2

def reorder_camera_corners(corners):
    # Ensure corners are ordered as: down-right, up-right, up-left, down-left
    sorted_corners = sorted(corners, key=lambda x: (x[1], x[0]))
    if sorted_corners[0][0] < sorted_corners[1][0]:
        down_right = sorted_corners[0]
        down_left = sorted_corners[1]
    else:
        down_right = sorted_corners[1]
        down_left = sorted_corners[0]
    if sorted_corners[2][0] < sorted_corners[3][0]:
        up_right = sorted_corners[2]
        up_left = sorted_corners[3]
    else:
        up_right = sorted_corners[3]
        up_left = sorted_corners[2]
    return np.array([down_right, up_right, up_left, down_left])

import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

def compute_camera_target_transformation(image_path, dv, dh):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path} during camera-target transformation calculation")
    
    # Determine the corresponding JSON file path
    pair_dir = os.path.dirname(image_path)
    json_path = os.path.join(pair_dir, "T_Chessboard_in_Camera.json")

    if not os.path.exists(json_path):
        raise ValueError(f"JSON file not found: {json_path}")

    # Load rvecs and tvecs from JSON file
    with open(json_path, 'r') as json_file:
        data = json.load(json_file)
        rvecs = np.array(data["rotation_vector"], dtype=np.float64).reshape(3, 1)
        tvecs = np.array(data["translation_vector"], dtype=np.float64).reshape(3, 1)

    # Convert rotation vector to rotation matrix
    rmat, _ = cv2.Rodrigues(rvecs)
    tvecs = tvecs.flatten()

    # Compute the homogeneous transformation matrix from the camera to the chessboard frame
    T_intern_in_cam = np.eye(4)
    T_intern_in_cam[:3, :3] = rmat
    T_intern_in_cam[:3, 3] = tvecs
    
    # Compute the homogeneous transformation matrix from the chessboard to the target frame
    T_target_in_intern = np.eye(4)
    T_target_in_intern[:3, 3] = [-dv, -dh, 0]

    T_target_in_camera = T_intern_in_cam @ T_target_in_intern

    # Compute Euler angles (ZYX convention) and translation distances
    rotation_matrix = T_target_in_camera[:3, :3]
    translation_vector = T_target_in_camera[:3, 3]
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('zyx', degrees=True)

    # Save to JSON file
    result_data = {
        "transformation_matrix": T_target_in_camera.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_vector": translation_vector.tolist()
    }

    json_output_path = os.path.join(pair_dir, "T_Camera_B.json")
    with open(json_output_path, 'w') as json_file:
        json.dump(result_data, json_file, indent=4)

    return T_target_in_camera
    
def prompt_user_for_exclusion():
    while True:
        user_input = input("Press 'N' to keep all points, 'ABCD' to remove all points and delete the pair, or any combination of 'A', 'B', 'C', 'D' to exclude specific points: ").strip().upper()
        if all(char in "ABCDN" for char in user_input):
            return user_input
        else:
            print("Invalid input. Please enter a valid combination of 'A', 'B', 'C', 'D', or 'N'.")


def calculate_inverse_transformation(T):
    R = T[:3, :3]
    t = T[:3, 3]
    R_inv = R.T
    t_inv = -R.T @ t
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = t_inv
    return T_inv

def save_transformation(transformation, file_path):
    with open(file_path, "w") as f:
        json.dump(transformation.tolist(), f)
    print(f"Transformation matrix saved to {file_path}")

import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def load_transformation(file_path):
    with open(file_path, "r") as f:
        transformation = np.array(json.load(f))
    return transformation

def calculate_camera_lidar_transformation(T_Camera_B, T_B_Lidar):
    return T_Camera_B @ T_B_Lidar

def extract_euler_angles_and_translation(T):
    rotation_matrix = T[:3, :3]
    translation_vector = T[:3, 3]
    
    # Extract Euler angles
    rotation = R.from_matrix(rotation_matrix)
    euler_angles = rotation.as_euler('zyx', degrees=True)
    
    translation_distances = translation_vector
    
    return euler_angles, translation_distances

def save_camera_lidar_transformation(pair_dir, T_Camera_Lidar, euler_angles, translation_distances):
    # Extract the index i from the subdirectory name pair_i
    pair_index = os.path.basename(pair_dir).split('_')[-1]

    data = {
        "transformation_matrix": T_Camera_Lidar.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_distances": translation_distances.tolist()
    }

    # Define the file name with the index i
    file_name = f"T_Camera_Lidar_{pair_index}.json"
    file_path = os.path.join(pair_dir, file_name)

    # Save the JSON file in the pair directory
    with open(file_path, "w") as f:
        json.dump(data, f)
    print(f"Camera-LiDAR transformation matrix, Euler angles, and translation distances saved to {file_path}")

    # Define the destination directory
    destination_dir = os.path.join(os.path.dirname(__file__), "T_Camera-Lidar")

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Copy the JSON file to the destination directory
    destination_path = os.path.join(destination_dir, file_name)
    shutil.copy(file_path, destination_path)
    print(f"File copied to {destination_path}")


import os
import shutil
import json
import numpy as np
from scipy.spatial.transform import Rotation as R

def copy_and_rename_transformations(src_dir, dest_dir):
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    pair_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    all_euler_angles = []
    all_translation_distances = []

    for pair_dir in pair_dirs:
        src_path = os.path.join(src_dir, pair_dir, "T_Camera_Lidar.json")
        if os.path.exists(src_path):
            with open(src_path, "r") as f:
                data = json.load(f)
                T_Camera_Lidar = np.array(data["transformation_matrix"])
                euler_angles = np.array(data["euler_angles"])
                translation_distances = np.array(data["translation_distances"])
                
                all_euler_angles.append(euler_angles)
                all_translation_distances.append(translation_distances)

                dest_path = os.path.join(dest_dir, f"T_Camera_Lidar_{pair_dir.split('_')[-1]}.json")
                shutil.copy(src_path, dest_path)

    return np.array(all_euler_angles), np.array(all_translation_distances)

import numpy as np
from scipy.spatial.transform import Rotation as R

def mean_average_transformations(euler_angles_list, translation_distances_list):
    # Calculate the mean for each Euler angle and each translation distance
    mean_euler_angles = np.mean(euler_angles_list, axis=0)
    mean_translation_distances = np.mean(translation_distances_list, axis=0)

    # Convert mean Euler angles to a rotation matrix
    mean_rotation = R.from_euler('zyx', mean_euler_angles, degrees=True).as_matrix()
    
    # Construct the transformation matrix using the mean values
    T_mean = np.eye(4)
    T_mean[:3, :3] = mean_rotation
    T_mean[:3, 3] = mean_translation_distances
    
    return T_mean, mean_euler_angles, mean_translation_distances

def save_mean_transformation(directory, T_mean, euler_angles, translation_distances):
    data = {
        "transformation_matrix": T_mean.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_distances": translation_distances.tolist()
    }
    file_path = os.path.join(directory, "T_Camera_Lidar_mean.json")
    with open(file_path, "w") as f:
        json.dump(data, f)
    print(f"Mean Camera-LiDAR transformation matrix, Euler angles, and translation distances saved to {file_path}")

def median_average_transformations(euler_angles_list, translation_distances_list):
    # Calculate the median for each Euler angle and each translation distance
    median_euler_angles = np.median(euler_angles_list, axis=0)
    median_translation_distances = np.median(translation_distances_list, axis=0)

    # Convert median Euler angles to a rotation matrix
    median_rotation = R.from_euler('zyx', median_euler_angles, degrees=True).as_matrix()
    
    # Construct the transformation matrix using the median values
    T_median = np.eye(4)
    T_median[:3, :3] = median_rotation
    T_median[:3, 3] = median_translation_distances
    
    return T_median, median_euler_angles, median_translation_distances

def save_median_transformation(directory, T_median, euler_angles, translation_distances):
    data = {
        "transformation_matrix": T_median.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_distances": translation_distances.tolist()
    }
    file_path = os.path.join(directory, "T_Camera_Lidar_median.json")
    with open(file_path, "w") as f:
        json.dump(data, f)
    print(f"Median Camera-LiDAR transformation matrix, Euler angles, and translation distances saved to {file_path}")

from sklearn.linear_model import RANSACRegressor

def ransac_average_transformations(euler_angles_list, translation_distances_list):
    # Convert lists to numpy arrays
    euler_angles_array = np.array(euler_angles_list)
    translation_distances_array = np.array(translation_distances_list)
    
    # RANSAC for Euler angles
    ransac_euler = RANSACRegressor()
    ransac_euler.fit(np.arange(len(euler_angles_array)).reshape(-1, 1), euler_angles_array)
    ransac_euler_angles = ransac_euler.predict(np.array([[0]]))
    
    # RANSAC for translation distances
    ransac_translation = RANSACRegressor()
    ransac_translation.fit(np.arange(len(translation_distances_array)).reshape(-1, 1), translation_distances_array)
    ransac_translation_distances = ransac_translation.predict(np.array([[0]]))

    # Convert RANSAC Euler angles to a rotation matrix
    ransac_rotation = R.from_euler('zyx', ransac_euler_angles[0], degrees=True).as_matrix()
    
    # Construct the transformation matrix using the RANSAC values
    T_ransac = np.eye(4)
    T_ransac[:3, :3] = ransac_rotation
    T_ransac[:3, 3] = ransac_translation_distances[0]
    
    return T_ransac, ransac_euler_angles[0], ransac_translation_distances[0]

def save_ransac_transformation(directory, T_ransac, euler_angles, translation_distances):
    data = {
        "transformation_matrix": T_ransac.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_distances": translation_distances.tolist()
    }
    file_path = os.path.join(directory, "T_Camera_Lidar_ransac.json")
    with open(file_path, "w") as f:
        json.dump(data, f)
    print(f"RANSAC Camera-LiDAR transformation matrix, Euler angles, and translation distances saved to {file_path}")

import os
import json
import numpy as np

def load_corners(file_path):
    corners = {}
    with open(file_path, 'r') as f:
        for line in f:
            parts = line.split()
            label = parts[0]
            point = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            corners[label] = point
    return corners

def compute_reprojection_error(T, corners_lidar, corners_camera):
    corners_lidar_homogeneous = np.hstack((corners_lidar, np.ones((corners_lidar.shape[0], 1))))
    projected_corners_camera_homogeneous = T @ corners_lidar_homogeneous.T
    projected_corners_camera = (projected_corners_camera_homogeneous / projected_corners_camera_homogeneous[3, :])[:3, :].T
    errors = np.linalg.norm(projected_corners_camera - corners_camera, axis=1)
    mean_error = np.mean(errors)
    return mean_error

import numpy as np
import os
import json

def evaluate_transformation(pair_dir, T):
    pair_index = os.path.basename(pair_dir).split('_')[-1]
    lidar_file = os.path.join(pair_dir, f"corners_in_lidar_{pair_index}.txt")
    camera_file = os.path.join(pair_dir, f"corners_in_camera_{pair_index}.txt")
    
    print(f"Evaluating transformation for pair directory: {pair_dir}")
    print(f"LiDAR corners file path: {lidar_file}")
    print(f"Camera corners file path: {camera_file}")

    if os.path.exists(lidar_file) and os.path.exists(camera_file):
        print(f"Files found for pair {pair_index}")
        corners_lidar_dict = load_corners(lidar_file)
        corners_camera_dict = load_corners(camera_file)

        # Ensure that the points are matched correctly
        common_labels = set(corners_lidar_dict.keys()).intersection(corners_camera_dict.keys())
        if not common_labels:
            raise ValueError(f"No common points found between LiDAR and camera corners in pair {pair_dir}")

        corners_lidar = np.array([corners_lidar_dict[label] for label in common_labels])
        corners_camera = np.array([corners_camera_dict[label] for label in common_labels])

        mean_error = compute_reprojection_error(T, corners_lidar, corners_camera)
        print(f"Reprojection error for pair {pair_dir}: {mean_error}")
        return mean_error
    else:
        print(f"Files not found for pair {pair_index}:")
        if not os.path.exists(lidar_file):
            print(f"  LiDAR file not found: {lidar_file}")
        if not os.path.exists(camera_file):
            print(f"  Camera file not found: {camera_file}")
        return None

def save_evaluation_results(directory, method_name, average_error, T, euler_angles, translation_distances):
    result_dir = os.path.join(directory, f"T_Camera_Lidar_{method_name}.json")
    data = {
        "transformation_matrix": T.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_distances": translation_distances.tolist(),
        "average_reprojection_error": average_error
    }
    with open(result_dir, "w") as f:
        json.dump(data, f)
    print(f"{method_name.capitalize()} Camera-LiDAR transformation matrix, Euler angles, translation distances, and average reprojection error saved to {result_dir}")

def evaluate_all_transformations(base_dir, result_dir):
    methods = ["mean", "median", "ransac"]
    results = []
    
    pair_dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    for method in methods:
        file_path = os.path.join(result_dir, f"T_Camera_Lidar_{method}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            T = np.array(data["transformation_matrix"])
            euler_angles = np.array(data["euler_angles"])
            translation_distances = np.array(data["translation_distances"])
            
            print(f"Evaluating {method} transformation matrix from {file_path}")
            errors = []
            for pair_dir in pair_dirs:
                error = evaluate_transformation(pair_dir, T)
                if error is not None:
                    errors.append(error)
            average_error = np.mean(errors) if errors else None
            if average_error is not None:
                save_evaluation_results(result_dir, method, average_error, T, euler_angles, translation_distances)
                results.append((average_error, euler_angles, translation_distances))
            print(f"Average reprojection error for {method}: {average_error}")
    
    # Evaluate individual transformations
    for file_name in os.listdir(result_dir):
        if file_name.startswith("T_Camera_Lidar_") and file_name.endswith(".json") and not any(method in file_name for method in methods):
            file_path = os.path.join(result_dir, file_name)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    data = json.load(f)
                T = np.array(data["transformation_matrix"])
                euler_angles = np.array(data["euler_angles"])
                translation_distances = np.array(data["translation_distances"])
                
                print(f"Evaluating {file_name} transformation matrix from {file_path}")
                errors = []
                for pair_dir in pair_dirs:
                    error = evaluate_transformation(pair_dir, T)
                    if error is not None:
                        errors.append(error)
                average_error = np.mean(errors) if errors else None
                if average_error is not None:
                    save_evaluation_results(result_dir, file_name.split(".")[0].split("T_Camera_Lidar_")[-1], average_error, T, euler_angles, translation_distances)
                    results.append((average_error, euler_angles, translation_distances))
                print(f"Average reprojection error for {file_name}: {average_error}")
    
    return results


def load_transformation_matrix(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
        T = np.array(data["transformation_matrix"])
    return T

import numpy as np
from scipy.spatial.transform import Rotation as R

def calculate_weighted_mean_transformations(directory):
    euler_angles_list = []
    translation_distances_list = []
    reprojection_errors = []

    # Load data from JSON files
    for file_name in os.listdir(directory):
        if file_name.startswith("T_Camera_Lidar_") and file_name.endswith(".json"):
            # Exclude the mean, median, ransac, and weighted files
            if not any(method in file_name for method in ["mean", "median", "ransac", "weighted"]):
                file_path = os.path.join(directory, file_name)
                with open(file_path, "r") as f:
                    data = json.load(f)
                    euler_angles = np.array(data["euler_angles"])
                    translation_distances = np.array(data["translation_distances"])
                    reprojection_error = data.get("average_reprojection_error")

                    if reprojection_error is not None:
                        euler_angles_list.append(euler_angles)
                        translation_distances_list.append(translation_distances)
                        reprojection_errors.append(reprojection_error)

    if not reprojection_errors:
        raise ValueError("No valid reprojection errors found for calculating weighted mean transformation")

    # Convert lists to numpy arrays
    euler_angles_array = np.array(euler_angles_list)
    translation_distances_array = np.array(translation_distances_list)
    reprojection_errors_array = np.array(reprojection_errors)

    # Calculate weights as the inverse of the reprojection errors
    weights = 1 / reprojection_errors_array
    normalized_weights = weights / np.sum(weights)

    # Debug prints to check intermediate values
    print("Reprojection Errors:", reprojection_errors_array)
    print("Weights:", weights)
    print("Normalized Weights:", normalized_weights)

    # Calculate the weighted average of euler angles and translation distances
    weighted_euler_angles = np.average(euler_angles_array, axis=0, weights=normalized_weights)
    weighted_translation_distances = np.average(translation_distances_array, axis=0, weights=normalized_weights)

    # Debug prints to check weighted averages
    print("Weighted Euler Angles:", weighted_euler_angles)
    print("Weighted Translation Distances:", weighted_translation_distances)

    # Convert weighted Euler angles to a rotation matrix
    weighted_rotation_matrix = R.from_euler('zyx', weighted_euler_angles, degrees=True).as_matrix()

    # Construct the weighted transformation matrix
    T_weighted = np.eye(4)
    T_weighted[:3, :3] = weighted_rotation_matrix
    T_weighted[:3, 3] = weighted_translation_distances

    return T_weighted, weighted_euler_angles, weighted_translation_distances

def save_weighted_transformation(directory, T_weighted, euler_angles, translation_distances, average_error):
    data = {
        "transformation_matrix": T_weighted.tolist(),
        "euler_angles": euler_angles.tolist(),
        "translation_distances": translation_distances.tolist(),
        "average_reprojection_error": average_error
    }
    file_path = os.path.join(directory, "T_Camera_Lidar_weighted.json")
    with open(file_path, "w") as f:
        json.dump(data, f)
    print(f"Weighted Camera-LiDAR transformation matrix, Euler angles, translation distances, and average reprojection error saved to {file_path}")
    print(f"Average reprojection error for weighted mean: {average_error}")

"""
import numpy as np
import cv2

def project_3d_to_2d(points_3d, intrinsic_matrix, dist_coeffs):
    # dist_coeffs = np.zeros((4, 1))  # Assuming no distortion
    rvec = np.zeros((3, 1))  # No rotation
    tvec = np.zeros((3, 1))  # No translation

    points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, intrinsic_matrix, dist_coeffs)
    return points_2d.reshape(-1, 2)

import matplotlib.pyplot as plt
import cv2

def display_image_with_points(image_path, points_3d, intrinsic_matrix, dist_coef, labels):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Error loading image {image_path} during visualization")

    # Ensure points_3d is a numpy array of shape (n, 3)
    if points_3d.shape[1] != 3:
        raise ValueError("points_3d should have shape (n, 3)")

    points_2d = project_3d_to_2d(points_3d, intrinsic_matrix, dist_coef)

    fig, ax = plt.subplots()
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    for label, point in zip(labels, points_2d):
        ax.scatter(point[0], point[1], c='red', s=100, edgecolors='black')
        ax.text(point[0], point[1], label, color='yellow', fontsize=12, ha='right', va='bottom')

    plt.show()
"""
import os
def remove_points_from_files(pair_dir, points_to_remove):
    pair_index = os.path.basename(pair_dir).split('_')[-1]
    camera_file = os.path.join(pair_dir, f"corners_in_camera_{pair_index}.txt")
    lidar_file = os.path.join(pair_dir, f"corners_in_lidar_{pair_index}.txt")

    print(f"Removing points {points_to_remove} from {camera_file} and {lidar_file}")

    remove_points_from_file(camera_file, points_to_remove)
    remove_points_from_file(lidar_file, points_to_remove)


def remove_points_from_file(file_path, points_to_remove):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        label = line.split()[0]
        if label not in points_to_remove:
            new_lines.append(line)
        else:
            print(f"Removing {label} from {file_path}")

    with open(file_path, 'w') as f:
        f.writelines(new_lines)
    print(f"Updated file {file_path} by removing points: {points_to_remove}")
    print("New contents:", new_lines)  

import os
import json
import numpy as np

def find_best_transformation(directory):
    best_error = float('inf')
    best_transformation_data = None
    best_file_name = None

    for file_name in os.listdir(directory):
        if file_name.startswith("T_Camera_Lidar_") and file_name.endswith(".json"):
            file_path = os.path.join(directory, file_name)
            with open(file_path, "r") as f:
                data = json.load(f)
                reprojection_error = data.get("average_reprojection_error")

                if reprojection_error is not None and reprojection_error < best_error:
                    best_error = reprojection_error
                    best_transformation_data = data
                    best_file_name = file_name
    print(f"Best transformation found in {best_file_name} with average reprojection error of {best_error}")

    if best_transformation_data is not None:
        final_file_path = os.path.join(directory, "T_Camera_Lidar_Final.json")
        with open(final_file_path, "w") as f:
            json.dump(best_transformation_data, f)
        print(f"Best transformation saved to {final_file_path} with average reprojection error of {best_error}")
    else:
        print("No valid transformation found")

    return best_transformation_data, best_file_name, best_error

import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize

def calculate_reprojection_error_for_optimization(T, corners_lidar, corners_camera):
    corners_lidar_homogeneous = np.hstack((corners_lidar, np.ones((corners_lidar.shape[0], 1))))
    projected_corners_camera_homogeneous = T @ corners_lidar_homogeneous.T
    projected_corners_camera = (projected_corners_camera_homogeneous / projected_corners_camera_homogeneous[3, :])[:3, :].T
    errors = np.linalg.norm(projected_corners_camera - corners_camera, axis=1)
    mean_error = np.mean(errors)
    return mean_error

# Define the cost function for optimization
def optimization_cost_function(params, corners_lidar, corners_camera):
    yaw, pitch, roll, tx, ty, tz = params
    rotation_matrix = R.from_euler('zyx', [yaw, pitch, roll], degrees=True).as_matrix()
    T = np.eye(4)
    T[:3, :3] = rotation_matrix
    T[:3, 3] = [tx, ty, tz]
    return calculate_reprojection_error_for_optimization(T, corners_lidar, corners_camera)

def optimize_transformation(corners_lidar_list, corners_camera_list, initial_params):
    result = minimize(
        optimization_cost_function, 
        initial_params, 
        args=(np.vstack(corners_lidar_list), np.vstack(corners_camera_list)), 
        method='L-BFGS-B',
        options={'disp': True}
    )
    return result