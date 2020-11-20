import time
from typing import Union
from torch import nn, optim, Tensor, no_grad
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import device as Device
import numpy as np
from .dataset import Salicon

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        dataset_root: str,
        summary_writer: SummaryWriter,
        device: Device,
        batch_size : int = 128 
    ):
        # load train/test splits of SALICON dataset
        train_dataset = Salicon(
            dataset_root + "train.pkl"
        )
        test_dataset = Salicon(
            dataset_root + "val.pkl"
        )

        # NEED TO ADD RANDOM HORIZONTAL FLIPPING (according to paper)
        # transformList = [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
        # transform = transforms.Compose(transformList)
        
        self.train_loader = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            pin_memory=True,
            num_workers=1,
        )
        self.val_loader = DataLoader(
            test_dataset,
            shuffle=False,
            batch_size=batch_size,
            num_workers=1,
            pin_memory=True,
        )
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = SGD(self.model.parameters(),lr=0.03, momentum=0.9, nesterov=True) 
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        self.model.train()
        for epoch in range(start_epoch, epochs):
            for batch, labels in self.train_loader:

                # load batch to device
                batch = batch.to(self.device)
                labels = labels.to(self.device)

                # train step
                step_start_time = time.time()
                logits = self.model.forward(batch)
                loss = self.criterion(logits,labels)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # log step
                if ((self.step + 1) % log_frequency) == 0:
                    with no_grad():
                        accuracy = compute_accuracy(labels, logits)
                    step_time = time.time() - step_start_time
                    self.log_metrics(epoch, accuracy, loss, step_time)
                    self.print_metrics(epoch, accuracy, loss, step_time)

                # need to update learning rate between 0.03 and 0.0001 (according to paper)
                # tilo says to use weight decay of 0.0005 (dunno what that means)
                # tilo also says dont worry about momentum decay (nice)

                # count steps
                self.step += 1

            # log epoch 
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # validate
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.model.train()

    def print_metrics(self, epoch, accuracy, loss, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"batch accuracy: {accuracy * 100:2.2f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, accuracy, loss, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)
        self.summary_writer.add_scalars(
                "accuracy",
                {"train": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "labels": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with no_grad():
            for batch, labels in self.val_loader:
                batch = batch.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(batch)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().numpy()
                results["preds"].extend(list(preds))
                results["labels"].extend(list(labels.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["labels"]), np.array(results["preds"])
        )
        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "accuracy",
                {"test": accuracy},
                self.step
        )
        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}, accuracy: {accuracy * 100:2.2f}")



def compute_accuracy(
    labels: Union[Tensor, np.ndarray], preds: Union[Tensor, np.ndarray]
) -> float:
    """
    Args:
        labels: ``(batch_size, class_count)`` tensor or array containing example labels
        preds: ``(batch_size, class_count)`` tensor or array containing model prediction
    """
   
    return 0


    


