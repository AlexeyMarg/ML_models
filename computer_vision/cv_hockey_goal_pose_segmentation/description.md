# Hockey Goal Detection and Geometry Estimation

Dataset: https://drive.google.com/drive/folders/1X0DBl1SeDW0JLYOwi8S33tH9TP2bmTxv?usp=sharing

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

### prepare_pose_dataset.py

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

    python prepare_pose_dataset.py --src label_studio_pose_export --dst yolo_pose_dataset --train-ratio 0.8

### prepare_seg_dataset.py

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

    python prepare_seg_dataset.py --src label_studio_seg_export --dst yolo_seg_dataset --train-ratio 0.8

### train_pose.py

This script trains a YOLO pose model.

It performs the following actions:

- loads a pretrained YOLO pose model;
- reads the prepared pose dataset configuration;
- trains the model on the prepared dataset;
- saves training results, logs, and weights;
- stores the best model weights for later inference.

Typical output:

    runs/pose/train/weights/best.pt

### train_seg.py

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








