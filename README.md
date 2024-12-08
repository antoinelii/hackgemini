# Segment agricultural parcels from satellite times series (Capgemini x Mines de Paris Hackathon 2024)

## Overview
Welcome to the Capgemini Invent 2024 Data Challenge!

We are excited to have you join this year's data challenge. For this new edition, you are asked to tackle a real-world problem using satellite imagery.

At Capgemini Invent, we have already demonstrated the power of satellite image analysis by detecting undeclared swimming pools and classifying agricultural parcels for government programs. By leveraging data-driven insights, businesses and authorities can unlock new opportunities and enhance operational efficiencies.

## The Challenge

In this competition, you will face the task of segmenting time series of satellite images to predict the type of crop being grown on various agricultural parcels. The time series data reflects changes in crop appearance throughout the growing season, and your model will need to capture these dynamics to produce accurate predictions.

This challenge will test your ability to work with time series data, satellite imagery, and geospatial analytics, offering a unique and meaningful opportunity to contribute to an important cause while honing your technical skills.

## Data
Input: sequences of T satellite images, centered on ~1km x ~1km zones.

    Each image is of shape (10, 128, 128) (channel-first convention), i.e. the images are (128, 128) and have 10 channels.
    Within a sequence, the images are already centered on the same point. Within a sequence, pixels at the same location point to the same parcels.
    Due to external factors, some sequences have more images than others, i.e. T varies.

Output: The ground truth (the target) is a semantic segmentation mask of the crop types in the images (20 classes)

Key resources

    Kaggle competition: https://www.kaggle.com/competitions/data-challenge-invent-mines-2024
