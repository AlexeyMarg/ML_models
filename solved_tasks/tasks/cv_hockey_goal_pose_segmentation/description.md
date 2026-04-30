# Hockey Goal Detection and Geometry Estimation

## Project Description

This project is designed for automatic detection of hockey goals and estimation of their geometric structure from images.

The main objective is to identify the hockey goal in an image and recover its geometry. Two neural-network-based approaches are used:

1. **Pose estimation** for detecting the goal and estimating its four corner points.
2. **Segmentation** for detecting the goal silhouette as a polygon.

The project includes scripts for preparing annotated data, training YOLO models, and testing trained models on individual images.

## Annotation Source

The dataset is annotated in Label Studio.

The original annotation project contains three types of annotations for each hockey goal:

- bounding box around the goal;
- four keypoints corresponding to the goal corners;
- polygon describing the visible goal silhouette.

The source data is exported from Label Studio using the "YOLO with Images" option.

Before training, the exported data is converted into separate datasets:

- one dataset for pose estimation;
- one dataset for segmentation.

The original Label Studio export is not edited manually.

## Source Data Structure

After export from Label Studio, the source data is expected to have the following structure:

    source_dataset/
      images/
        image_001.jpg
        image_002.jpg
        ...
      labels/
        image_001.txt
        image_002.txt
        ...

Each image must have a corresponding label file with the same filename and the `.txt` extension.

For pose estimation, the source labels contain object bounding boxes and four corner keypoints.

For segmentation, the source labels contain polygon annotations of the goal silhouette.

## Pose Estimation Task

The pose estimation model is trained to detect the hockey goal and estimate four ordered keypoints:

- top-left corner;
- top-right corner;
- bottom-left corner;
- bottom-right corner.

This approach is the main one for direct corner localization.

The pose model output includes:

- detected goal bounding box;
- four ordered goal corner points;
- confidence score.

## Segmentation Task

The segmentation model is trained to detect the visible silhouette of the hockey goal.

The segmentation output is a mask or polygonal contour. Since the segmentation model may return a contour with many points, an additional post-processing step is used to approximate the contour with a four-point quadrilateral.

The segmentation pipeline is:

    image -> segmentation model -> mask -> contour extraction -> quadrilateral approximation

This approach is useful for estimating the general shape of the goal and can be used as an additional source of geometric information.

## Project Files

### prepare_goal_pose_dataset.py

This script prepares the pose-estimation dataset.

It performs the following actions:

- reads the exported Label Studio dataset;
- checks that each image has a corresponding label file;
- validates pose annotations;
- splits the dataset into training and validation subsets;
- copies images and labels into the required training structure;
- creates a dataset configuration file for YOLO pose training.

Input:

    label_studio_pose_export/

Output:

    yolo_pose_dataset/

Typical usage:

    python prepare_goal_pose_dataset.py --src label_studio_pose_export --dst yolo_pose_dataset --train-ratio 0.8

### prepare_goal_seg_dataset.py

This script prepares the segmentation dataset.

It performs the following actions:

- reads the exported Label Studio segmentation dataset;
- checks that each image has a corresponding label file;
- validates polygon annotations;
- splits the dataset into training and validation subsets;
- copies images and labels into the required training structure;
- creates a dataset configuration file for YOLO segmentation training.

Input:

    label_studio_seg_export/

Output:

    yolo_seg_dataset/

Typical usage:

    python prepare_goal_seg_dataset.py --src label_studio_seg_export --dst yolo_seg_dataset --train-ratio 0.8

### train_goal_pose.py

This script trains a YOLO pose model.

It performs the following actions:

- loads a pretrained YOLO pose model;
- reads the prepared pose dataset configuration;
- trains the model on the prepared dataset;
- saves training results, logs, and weights;
- stores the best model weights for later inference.

Typical output:

    runs/pose/train/weights/best.pt

### train_goal_seg.py

This script trains a YOLO segmentation model.

It performs the following actions:

- loads a pretrained YOLO segmentation model;
- reads the prepared segmentation dataset configuration;
- trains the model on the prepared dataset;
- saves training results, logs, and weights;
- stores the best model weights for later inference.

Typical output:

    runs/segment/train/weights/best.pt

### predict_one_pose.py

This script applies the trained pose model to a single test image.

It performs the following actions:

- loads the trained pose model;
- reads one input image;
- detects the hockey goal;
- draws the predicted bounding box;
- draws the four predicted corner points;
- saves the visualization result.

Input:

    test image

Output:

    image with detected goal and predicted keypoints

### predict_one_seg.py

This script applies the trained segmentation model to a single test image.

It performs the following actions:

- loads the trained segmentation model;
- reads one input image;
- detects the hockey goal silhouette;
- extracts the predicted mask contour;
- approximates the contour with four points;
- draws the resulting quadrilateral;
- saves the visualization result.

Input:

    test image

Output:

    image with detected goal silhouette and approximated quadrilateral

## Recommended Workflow

### 1. Annotate images in Label Studio

Each hockey goal should be annotated with:

- bounding box;
- four corner keypoints;
- polygon silhouette.

### 2. Export annotations

Export the data from Label Studio using the appropriate export option with images.

Create separate exports if needed:

- one export for pose annotations;
- one export for segmentation annotations.

### 3. Prepare datasets

For pose estimation:

    python prepare_goal_pose_dataset.py --src pose_dataset --dst yolo_pose_dataset --train-ratio 0.8

For segmentation:

    python prepare_goal_seg_dataset.py --src seg_dataset --dst yolo_seg_dataset --train-ratio 0.8

### 4. Train models

Train the pose model:

    python train_goal_pose.py

Train the segmentation model:

    python train_goal_seg.py

### 5. Test models

Run pose prediction on one image:

    python predict_one_pose.py

Run segmentation prediction on one image:

    python predict_one_seg.py

## Data Preparation Notes

The dataset preparation scripts do not modify the original Label Studio export.

They only create a new training dataset directory with:

- training images;
- validation images;
- training labels;
- validation labels;
- dataset configuration file.

This makes it possible to keep the original exported annotations unchanged and repeat the preparation step with different train-validation splits if needed.

## Practical Notes

The pose model is preferred when the main goal is accurate localization of the four goal corners.

The segmentation model is useful when the goal silhouette is important or when the goal shape should be estimated as a region.

For robust performance, the dataset should contain images with:

- different viewpoints;
- different camera distances;
- partially occluded goals;
- goalkeepers and players in front of the goal;
- different lighting conditions;
- motion blur;
- different arenas and backgrounds.

## Summary

This project provides a complete workflow for hockey goal geometry estimation:

- annotation in Label Studio;
- preparation of pose and segmentation datasets;
- training of YOLO pose and segmentation models;
- inference on test images;
- visualization of detected goals, keypoints, and polygonal silhouettes.

The pose pipeline is used for direct estimation of four goal corners, while the segmentation pipeline provides an additional geometric representation of the goal silhouette.







https://drive.google.com/drive/folders/1X0DBl1SeDW0JLYOwi8S33tH9TP2bmTxv?usp=sharing