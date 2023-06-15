from torchvision import models
import torch
import torch.nn as nn


def restnet50_transfer_learning():
    """
    RestNet50 Transfer Learning
    :return:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = models.resnet50(weights="IMAGENET1K_V1").to(device)
    for param in model.parameters():
        param.requires_grad = False
        model.fc = nn.Sequential(
            nn.Linear(2048, 128), nn.ReLU(inplace=True), nn.Linear(128, 2)
        ).to(device)
    return model
