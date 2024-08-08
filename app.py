# app.py
import os
import shutil
import json
import numpy as np
import cv2
import streamlit as st
from utils import (
    filter_and_save_images, upload_images, process_images, detect_and_cluster, select_elite_images,
    final_calibration, calculate_mean_reprojection_error_all, calculate_mean_reprojection_error_elites, detect_image_resolution,
    approximate_intrinsic_parameters, match_point_clouds_with_images, save_pairs, process_pair,
    visualize_and_label_point_clouds, compute_camera_target_transformation, display_camera_target_visualization,
    calculate_inverse_transformation, save_transformation, load_transformation, calculate_camera_lidar_transformation, extract_euler_angles_and_translation,
    save_camera_lidar_transformation, mean_average_transformations, save_mean_transformation,
    median_average_transformations, save_median_transformation, ransac_average_transformations,
    save_ransac_transformation, evaluate_all_transformations, save_weighted_transformation, calculate_weighted_mean_transformations, evaluate_transformation,
    find_best_transformation, optimize_transformation, calculate_reprojection_error_for_optimization, save_camera_lidar_transformation, save_camera_lidar_transformation, load_corners
)
import logging
from logging import getLogger

import streamlit as st




# Create necessary directories and ensure they are empty if they exist
directories = ["uploaded_images_0", "uploaded_images", "invalid_images"]
for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)

# Create necessary directories and ensure they are empty if they exist
directories = ["uploaded_images", "undistorted_images", "clusters", "elites", "uploaded_point_clouds", "lidar-camera_pairs", "T_Camera-Lidar"]
for directory in directories:
    if os.path.exists(directory):
        # Clear the directory contents
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    else:
        os.makedirs(directory)

st.title("Camera and LiDAR Calibration Dashboard")

# User selects the number of normal cameras
num_normal_cameras = st.number_input("Number of normal cameras", min_value=1, max_value=8, value=1)

if num_normal_cameras > 1:
    st.error("Multiple normal cameras are not yet supported.")

# User selects to add a panoramic camera
add_panoramic_camera = st.checkbox("Add panoramic camera")
if add_panoramic_camera:
    st.error("Panoramic cameras are not supported yet.")

# User selects to add a LiDAR
add_lidar = st.checkbox("Add LiDAR")

uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
print("uploaded_files", uploaded_files)

if add_lidar:
    uploaded_point_clouds = st.file_uploader("Upload Point Clouds", type=["txt"], accept_multiple_files=True)
else:
    uploaded_point_clouds = None

# User specifies chessboard parameters
st.subheader("Chessboard Parameters")
squares_x = st.number_input("Number of squares horizontally", min_value=1, value=8) - 1  # -1 to account for internal corners
squares_y = st.number_input("Number of squares vertically", min_value=1, value=7) - 1  # -1 to account for internal corners
square_size = st.number_input("Size of each square (mm)", min_value=1.0, value=107.0) / 1000.0  # Convert to meters

# User specifies distances from the chessboard to the target edges
st.subheader("Target to Chessboard Edge Distances")
distance_vertical_edge = st.number_input("Distance from the extreme internal corner to the nearest vertical edge of the target (mm)", min_value=1, value=215) / 1000.0  # Convert to meters
distance_horizontal_edge = st.number_input("Distance from the extreme internal corner to the nearest horizontal edge of the target (mm)", min_value=1, value=215) / 1000.0  # Convert to meters

# User selects which distortion coefficients to correct
st.subheader("Select Distortion Coefficients to Correct")

st.markdown("**Radial Distortion Coefficients:**")
col1, col2, col3 = st.columns(3)
with col1:
    k1 = st.checkbox("k1")
with col2:
    k2 = st.checkbox("k2")
with col3:
    k3 = st.checkbox("k3")

st.markdown("**Tangential Distortion Coefficients:**")
col4, col5 = st.columns(2)
with col4:
    p1 = st.checkbox("p1")
with col5:
    p2 = st.checkbox("p2")

# User selects the number of clusters
st.subheader("Clustering Parameters")
num_clusters = st.number_input("Number of Clusters", min_value=1, value=27)

# User selects whether to calculate RMSE
calculate_rmse = st.checkbox("Calculate RMSE")

if st.button("Process Images"):
    if uploaded_files:
        images = []
        image_paths = []
        for uploaded_file in uploaded_files:
            image_path = os.path.join("uploaded_images_0", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            image_paths.append(image_path)

        st.write("Images uploaded successfully.")

        valid_image_paths, invalid_image_paths = filter_and_save_images(
            image_paths, squares_x, squares_y, "uploaded_images", "invalid_images"
        )
        
        st.write(f"Valid images: {len(valid_image_paths)}")
        st.write(f"Invalid images: {len(invalid_image_paths)}")

        if valid_image_paths:
            st.write("Valid images saved to 'uploaded_images' directory.")
        if invalid_image_paths:
            st.write("Invalid images saved to 'invalid_images' directory.")
        images = [cv2.imread(img_path) for img_path in valid_image_paths]
        # Detect resolution of the first image to approximate intrinsic parameters
        width, height = detect_image_resolution(image_paths[0])
        intrinsic_matrix = approximate_intrinsic_parameters(width, height)

        # Initial calibration using the approximated intrinsic parameters
        undistorted_images, intrinsic_matrix = process_images(
            images, 
            [k1, k2, k3], 
            [p1, p2], 
            intrinsic_matrix,
            squares_x,
            squares_y,
            square_size
        )
        st.write("Image distortion correction completed.")

        # Save undistorted images and maintain mapping
        undistorted_paths = []
        for idx, img in enumerate(undistorted_images):
            undistorted_path = os.path.join("undistorted_images", os.path.basename(image_paths[idx]))
            cv2.imwrite(undistorted_path, img)
            undistorted_paths.append(undistorted_path)

        st.write("Undistorted images saved successfully.")

        features = detect_and_cluster(undistorted_paths, intrinsic_matrix, squares_x, squares_y, square_size, num_clusters)
        st.write("Clustering completed.")

        # Save clustered images
        if os.path.exists("clusters"):
            shutil.rmtree("clusters")
        os.makedirs("clusters")
        for cluster_id, img_paths in features.items():
            cluster_dir = os.path.join("clusters", f"cluster_{cluster_id}")
            os.makedirs(cluster_dir)
            for img_path in img_paths:
                shutil.copy(img_path, cluster_dir)

        st.write("Clustered images saved successfully.")

        elite_images = select_elite_images(features, intrinsic_matrix, squares_x, squares_y, square_size)
        st.write("Elite image selection completed.")

        # Save elite images
        if os.path.exists("elites"):
            shutil.rmtree("elites")
        os.makedirs("elites")
        for img_path in elite_images:
            shutil.copy(img_path, "elites")

        st.write("Elite images saved successfully.")

        mtx, dist, rvecs, tvecs, objpoints, imgpoints = final_calibration(elite_images, intrinsic_matrix, squares_x, squares_y, square_size)
        st.write("Final calibration completed.")
        st.write("Intrinsic matrix:")
        st.write(mtx)
        st.write("Distortion coefficients:")
        st.write(dist)
        st.write("Rotation vectors:")
        st.write(rvecs)
        st.write("Translation vectors:")
        st.write(tvecs)
        
        mean_reprojection_error_elites, rmse_elites = calculate_mean_reprojection_error_elites(
            elite_images, mtx, dist, rvecs, tvecs, squares_x, squares_y, square_size
        )
        
        mean_reprojection_error_all, rmse_all = calculate_mean_reprojection_error_all(
            undistorted_paths, mtx, dist, squares_x, squares_y, square_size
        )
        
        st.write("Mean reprojection errors calculated.")

        # Save results to results.json
        results = {
            "intrinsic_matrix": mtx.tolist(),
            "distortion_coefficients": dist.tolist(),
            "mean_reprojection_error_all": mean_reprojection_error_all,
            "mean_reprojection_error_elites": mean_reprojection_error_elites
        }
        if calculate_rmse:
            results["rmse_all"] = rmse_all
            results["rmse_elites"] = rmse_elites

        with open("results.json", "w") as f:
            json.dump(results, f)

        st.write("Results saved to results.json.")
        st.write("Processing complete. Check the results below.")

        # Display results on the dashboard
        st.header("Intrinsic Camera Calibration Results")
        st.subheader("Intrinsic Matrix")
        st.write(np.array(results["intrinsic_matrix"]))
        st.subheader("Distortion Coefficients")
        st.write(np.array(results["distortion_coefficients"]))
        st.subheader("Mean Reprojection Error (All Images)")
        st.write(results["mean_reprojection_error_all"])
        if calculate_rmse:
            st.subheader("RMSE (All Images)")
            st.write(results["rmse_all"])
        st.subheader("Mean Reprojection Error (Elite Images)")
        st.write(results["mean_reprojection_error_elites"])
        if calculate_rmse:
            st.subheader("RMSE (Elite Images)")
            st.write(results["rmse_elites"])

        if add_lidar and uploaded_point_clouds:
            st.write("Starting Camera-LiDAR calibration...")

            # Save uploaded point clouds
            point_cloud_paths = []
            for uploaded_file in uploaded_point_clouds:
                point_cloud_path = os.path.join("uploaded_point_clouds", uploaded_file.name)
                with open(point_cloud_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                point_cloud_paths.append(point_cloud_path)

            st.write("Point clouds uploaded successfully.")
            st.write("Uploaded point clouds:")
            st.write(point_cloud_paths)  # Log the uploaded point cloud files

            # Match point clouds with elite images
            point_cloud_paths = [os.path.join("uploaded_point_clouds", os.path.basename(pc)) for pc in point_cloud_paths]
            image_paths_full = [os.path.join("elites", os.path.basename(img)) for img in elite_images]
            matched_pairs = match_point_clouds_with_images(image_paths_full, point_cloud_paths)
            
            st.write("Matched pairs:")
            st.write(matched_pairs)  # Log the uploaded point cloud files

            # Save matched pairs
            save_pairs(matched_pairs, "lidar-camera_pairs")

            st.write("Lidar-camera pairs saved successfully.")
            
            # Visualize and label point clouds
            visualize_and_label_point_clouds(mtx, dist, squares_x, squares_y, square_size, distance_vertical_edge, distance_horizontal_edge)
            
            st.write("Camera-LiDAR calibrated successfully.")

            # Collect all the transformations
            euler_angles_list = []
            translation_distances_list = []
            for file_name in os.listdir("T_Camera-Lidar"):
                if file_name.startswith("T_Camera_Lidar_") and file_name.endswith(".json"):
                    with open(os.path.join("T_Camera-Lidar", file_name), "r") as f:
                        data = json.load(f)
                        euler_angles_list.append(data["euler_angles"])
                        translation_distances_list.append(data["translation_distances"])

            # Calculate the mean transformation
            if euler_angles_list and translation_distances_list:
                T_mean, mean_euler_angles, mean_translation_distances = mean_average_transformations(euler_angles_list, translation_distances_list)
                save_mean_transformation("T_Camera-Lidar", T_mean, mean_euler_angles, mean_translation_distances)
                st.write("Mean Camera-LiDAR transformation calculated and saved.")

                # Calculate the median transformation
                T_median, median_euler_angles, median_translation_distances = median_average_transformations(euler_angles_list, translation_distances_list)
                save_median_transformation("T_Camera-Lidar", T_median, median_euler_angles, median_translation_distances)
                st.write("Median Camera-LiDAR transformation calculated and saved.")

                # Calculate the RANSAC transformation
                T_ransac, ransac_euler_angles, ransac_translation_distances = ransac_average_transformations(euler_angles_list, translation_distances_list)
                save_ransac_transformation("T_Camera-Lidar", T_ransac, ransac_euler_angles, ransac_translation_distances)
                st.write("RANSAC Camera-LiDAR transformation calculated and saved.")
            else:
                st.error("No Euler angles or translation distances found for computing average transformations.")
            
            pair_dirs= os.listdir("lidar-camera_pairs")

            # Evaluate all transformations
            base_dir = os.path.join(os.getcwd(), "lidar-camera_pairs")
            result_dir = os.path.join(os.getcwd(), "T_Camera-Lidar")
            evaluate_all_transformations(base_dir, result_dir)        
            st.write("All transformations evaluated.")

            # Calculate and save the weighted mean transformation
            T_weighted, weighted_euler_angles, weighted_translation_distances = calculate_weighted_mean_transformations("T_Camera-Lidar")

            # Initialize a variable to store cumulative weighted error
            total_weighted_error = 0

            # Base directory for pairs
            base_pair_dir = "lidar-camera_pairs"

            # Iterate over each pair directory and evaluate the transformation
            for pair_name in pair_dirs:
                pair_dir = os.path.join(base_pair_dir, pair_name)
                error = evaluate_transformation(pair_dir, T_weighted)
                print(f"Error for {pair_dir}: {error}")
                if error is not None:
                    total_weighted_error += error

            # Calculate the weighted average error
            weighted_average_error = total_weighted_error / len(pair_dirs)

            save_weighted_transformation("T_Camera-Lidar", T_weighted, weighted_euler_angles, weighted_translation_distances, weighted_average_error)

            # Display reprojection errors
            st.header("Reprojection Errors for Camera-LiDAR Transformations")
            methods = ["mean", "median", "ransac", "weighted"]
            for method in methods:
                file_path = os.path.join("T_Camera-Lidar", f"T_Camera_Lidar_{method}.json")
                if os.path.exists(file_path):
                    with open(file_path, "r") as f:
                        data = json.load(f)
                    st.subheader(f"Reprojection Error for {method.capitalize()} Transformation")
                    st.write(data.get("average_reprojection_error", "N/A"))

            # Display reprojection errors for individual transformations
            for file_name in os.listdir("T_Camera-Lidar"):
                if file_name.startswith("T_Camera_Lidar_") and file_name.endswith(".json") and not any(method in file_name for method in methods):
                    file_path = os.path.join("T_Camera-Lidar", file_name)
                    if os.path.exists(file_path):
                        with open(file_path, "r") as f:
                            data = json.load(f)
                        st.subheader(f"Reprojection Error for {file_name.split('.')[0]}")
                        st.write(data.get("average_reprojection_error", "N/A"))

            # Find and display the best transformation
            best_transformation_data, best_file_name, best_error = find_best_transformation("T_Camera-Lidar")

            if best_transformation_data:
                st.header("Best Camera-LiDAR Transformation")
                st.subheader("Transformation Matrix")
                st.write(np.array(best_transformation_data["transformation_matrix"]))
                st.subheader("Euler Angles (Degrees)")
                st.write(np.array(best_transformation_data["euler_angles"]))
                st.subheader("Translation Distances (Meters)")
                st.write(np.array(best_transformation_data["translation_distances"]))
                st.subheader("Average Reprojection Error")
                st.write(best_error)
            else:
                st.error("No valid transformation found")

            # Optimization step
            st.write("Starting optimization...")
            initial_params = best_transformation_data["euler_angles"] + best_transformation_data["translation_distances"]

            base_pair_dir = "lidar-camera_pairs"
            corners_lidar_list = []
            corners_camera_list = []

            for pair_name in pair_dirs:
                pair_dir = os.path.join(base_pair_dir, pair_name)
                pair_index = os.path.basename(pair_dir).split('_')[-1]
                corners_lidar_file = os.path.join(pair_dir, f"corners_in_lidar_{pair_index}.txt")
                corners_camera_file = os.path.join(pair_dir, f"corners_in_camera_{pair_index}.txt")

                print(f"Checking pair directory: {pair_dir}")
                print(f"LiDAR corners file: {corners_lidar_file}")
                print(f"Camera corners file: {corners_camera_file}")

                if os.path.exists(corners_lidar_file) and os.path.exists(corners_camera_file):
                    corners_lidar_dict = load_corners(corners_lidar_file)
                    corners_camera_dict = load_corners(corners_camera_file)

                    common_labels = set(corners_lidar_dict.keys()).intersection(corners_camera_dict.keys())
                    print(f"Common labels: {common_labels}")

                    if common_labels:
                        corners_lidar = np.array([corners_lidar_dict[label] for label in common_labels])
                        corners_camera = np.array([corners_camera_dict[label] for label in common_labels])

                        corners_lidar_list.append(corners_lidar)
                        corners_camera_list.append(corners_camera)
                    else:
                        print(f"No common labels found in {pair_dir}")
                else:
                    if not os.path.exists(corners_lidar_file):
                        print(f"LiDAR corners file not found: {corners_lidar_file}")
                    if not os.path.exists(corners_camera_file):
                        print(f"Camera corners file not found: {corners_camera_file}")

            if corners_lidar_list and corners_camera_list:
                result = optimize_transformation(corners_lidar_list, corners_camera_list, initial_params)
                optimized_params = result.x
                optimized_yaw, optimized_pitch, optimized_roll = optimized_params[:3]
                optimized_translation = optimized_params[3:]

                optimized_rotation_matrix = R.from_euler('zyx', [optimized_yaw, optimized_pitch, optimized_roll], degrees=True).as_matrix()
                T_optimized = np.eye(4)
                T_optimized[:3, :3] = optimized_rotation_matrix
                T_optimized[:3, 3] = optimized_translation

                optimized_error = calculate_reprojection_error_for_optimization(T_optimized, np.vstack(corners_lidar_list), np.vstack(corners_camera_list))

                optimized_data = {
                    "transformation_matrix": T_optimized.tolist(),
                    "euler_angles": [optimized_yaw, optimized_pitch, optimized_roll],
                    "translation_distances": optimized_translation.tolist(),
                    "average_reprojection_error": optimized_error
                }

                with open(os.path.join("T_Camera-Lidar", "T_Camera_Lidar_optimized.json"), "w") as f:
                    json.dump(optimized_data, f)

                st.header("Optimized Camera-LiDAR Transformation")
                st.subheader("Transformation Matrix")
                st.write(T_optimized)
                st.subheader("Euler Angles (Degrees)")
                st.write([optimized_yaw, optimized_pitch, optimized_roll])
                st.subheader("Translation Distances (Meters)")
                st.write(optimized_translation)
                st.subheader("Average Reprojection Error")
                st.write(optimized_error)
            else:
                st.error("No common points found for optimization")

    else:
        st.error("Please upload at least one image.")
