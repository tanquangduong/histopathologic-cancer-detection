import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from .input import get_data_file_path
from idetect.load_dataset import LoadCancerDataset


class CancerDetectTrainer:
    def __init__(self, trainPath, valPath, class_dict, config_path=None):
        if config_path:
            with open(os.path.join(os.getcwd(), config_path)) as f:
                self.parameters = json.load(f)
        else:
            with open(get_data_file_path("config/hyperparameter.json"), "rb") as f:
                self.parameters = json.load(f)

        transform_data_train = T.Compose(
            [T.Resize(224), T.RandomHorizontalFlip(), T.ToTensor()]
        )
        transform_data_val = T.Compose([T.Resize(224), T.ToTensor()])

        cancer_train_set = LoadCancerDataset(
            datafolder=trainPath,
            transform=transform_data_train,
            labels_dict=class_dict,
        )
        cancer_val_set = LoadCancerDataset(
            datafolder=valPath,
            transform=transform_data_val,
            labels_dict=class_dict,
        )

        cancer_train_dataloader = DataLoader(
            cancer_train_set,
            self.parameters["batch_size"],
            num_workers=self.parameters["num_workers"],
            pin_memory=self.parameters["pin_memory"],
            shuffle=self.parameters["shuffle"],
        )
        cancer_val_dataloader = DataLoader(
            cancer_val_set,
            self.parameters["batch_size"],
            num_workers=self.parameters["num_workers"],
            pin_memory=self.parameters["pin_memory"],
        )

        self.dataset_size = {
            "train": len(os.listdir(trainPath)),
            "validation": len(os.listdir(valPath)),
        }

        self.dataloaders = {
            "train": cancer_train_dataloader,
            "validation": cancer_val_dataloader,
        }

    def train_model(self, model):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.fc.parameters())

        for epoch in range(self.parameters["num_epochs"]):
            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 10)

            for phase in ["train", "validation"]:
                if phase == "train":
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in self.dataloaders[phase]:
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    _, preds = torch.max(outputs, 1)
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                epoch_loss = running_loss / self.dataset_size[phase]
                epoch_acc = running_corrects.double() / self.dataset_size[phase]

                print(
                    "{} loss: {:.4f}, acc: {:.4f}".format(phase, epoch_loss, epoch_acc)
                )
        return model
