"""
AUTHOR: Tan Quang Duong
Date: 6/15/2023
Purpose: Train a model to detect cancer from kaggle dataset: histopathologic-cancer-detection
"""
import os
import argparse
import pandas as pd
from idetect.load_model import restnet50_transfer_learning
from idetect.train_engine import CancerDetectTrainer
import torch

parser = argparse.ArgumentParser()
parser.add_argument(
    "--train_path",
    type=str,
    default="./datasets/cancer_train_dataset",
    help="path to train dataset directory",
)
parser.add_argument(
    "--val_path",
    type=str,
    default="./datasets/cancer_validation_dataset",
    help="path to validation dataset directory",
)
parser.add_argument(
    "--label_path",
    type=str,
    default="./datasets/selected_image_labels.csv",
    help="path to label dataset directory",
)
parser.add_argument(
    "--param_path", type=str, default=None, help="path to hyperparameter json file"
)
parser.add_argument(
    "--model_path",
    type=str,
    default="./models/pytorch",
    help="path to save trained model weights",
)
parser.add_argument(
    "--weight_name",
    type=str,
    default="weights.h5",
    help="name to save trained model weights",
)

if __name__ == "__main__":
    # Load parameters
    args, _ = parser.parse_known_args()
    trainPath = args.train_path
    valPath = args.val_path
    labelPath = args.label_path
    paramPath = args.param_path
    modelPath = args.model_path
    weightName = args.weight_name
    weightPath = os.path.join(modelPath, weightName)

    # Load label for training
    image_labels = pd.read_csv(labelPath)
    image_labels_dict = {k: v for k, v in zip(image_labels.id, image_labels.label)}

    # Load and train model
    MODEL = restnet50_transfer_learning()
    engine = CancerDetectTrainer(
        trainPath=trainPath,
        valPath=valPath,
        class_dict=image_labels_dict,
        config_path=paramPath,
    )
    TRAINED_MODEL = engine.train_model(MODEL)

    # Save trained model weight
    torch.save(TRAINED_MODEL.state_dict(), weightPath)
