"""
AUTHOR: Tan Quang Duong
Date: 6/15/2023
Purpose: Train a model to detect cancer from kaggle dataset: histopathologic-cancer-detection
"""
import os
import argparse
import pandas as pd
import torch
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
import torchvision.transforms as T
from idetect.load_model import restnet50_transfer_learning

parser = argparse.ArgumentParser()
parser.add_argument(
    "--test_image_path",
    type=str,
    default="./datasets/cancer_test_dataset/01ca4aacd7904afeba78403d162b3c0fef535944.tif",
    help="path to test dataset directory",
)
parser.add_argument(
    "--test_label_path",
    type=str,
    default="./datasets//test_image_labels.csv",
    help="path to label of test dataset as csv file",
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


def plot_save_prediction(img_list):
    fig, ax = plt.subplots(1, len(img_list), figsize=(20, 5))
    for i, img in enumerate(img_list):
#         ax = axs[i]
        ax.axis("off")
        ax.set_title(
            "{:.0f}% Nomal, {:.0f}% Cancer".format(
                100 * pred_probs[i, 0], 100 * pred_probs[i, 1]
            )
        )
        ax.imshow(img)
    fig.savefig("./output/prediction.png")


if __name__ == "__main__":
    args, _ = parser.parse_known_args()
    testImagePath = args.test_image_path
    testLabelPath = args.test_label_path
    modelPath = args.model_path
    weightName = args.weight_name
    weightPath = os.path.join(modelPath, weightName)

    # load device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load label for training
    test_labels = pd.read_csv(testLabelPath)

    # Load trained model weights
    MODEL = restnet50_transfer_learning()
    MODEL.load_state_dict(
        torch.load("models/pytorch/weights.h5", map_location=torch.device("cpu"))
    )

    # load image
    test_img_paths = [testImagePath]
    img_list = [Image.open(img_path) for img_path in test_img_paths]
    transform_test_image = T.Compose([T.Resize(224), T.ToTensor()])
    test_batch = torch.stack([transform_test_image(img).to(device) for img in img_list])

    # Inference
    pred_logits_tensor = MODEL(test_batch)
    pred_probs = F.softmax(pred_logits_tensor, dim=1).cpu().data.numpy()
    print(pred_probs)
    plot_save_prediction(img_list=img_list)
