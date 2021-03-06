import time
from typing import Union
from torch import nn, optim, Tensor, no_grad, save, mean, sqrt, sum as tsum
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
        batch_size : int = 128,
        cc_loss : bool = False
    ):
        # load train/test splits of SALICON dataset
        train_dataset = Salicon(
            dataset_root + "train.pkl"
        )
        test_dataset = Salicon(
            dataset_root + "val.pkl"
        )
        
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
        if cc_loss:
            self.criterion = CCLoss
        else:
            self.criterion = nn.MSELoss()
        self.optimizer = SGD(self.model.parameters(),lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True) 
        self.summary_writer = summary_writer
        self.step = 0

    def train(
        self,
        epochs: int,
        val_frequency: int,
        log_frequency: int = 5,
        start_epoch: int = 0
    ):
        lrs = np.linspace(0.03,0.0001,epochs)
        for epoch in range(start_epoch, epochs):
            self.model.train()
            for batch, gts in self.train_loader:
                # LR decay
                # need to update learning rate between 0.03 and 0.0001 (according to paper)
                optimstate = self.optimizer.state_dict()
                self.optimizer = SGD(self.model.parameters(),lr=lrs[epoch], momentum=0.9, weight_decay=0.0005, nesterov=True)
                self.optimizer.load_state_dict(optimstate)

                self.optimizer.zero_grad()
                # load batch to device
                batch = batch.to(self.device)
                gts = gts.to(self.device)

                # train step
                step_start_time = time.time()
                output = self.model.forward(batch)
                loss = self.criterion(output,gts)
                loss.backward()
                self.optimizer.step()

                # log step
                if ((self.step + 1) % log_frequency) == 0:
                    step_time = time.time() - step_start_time
                    self.log_metrics(epoch, loss, step_time)
                    self.print_metrics(epoch, loss, step_time)

                # count steps
                self.step += 1

            # log epoch 
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # validate
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.model.train()
            if (epoch+1) % 10 == 0:
                save(self.model,"checkp_model.pkl") 

    def print_metrics(self, epoch, loss, step_time):
        epoch_step = self.step % len(self.train_loader)
        print(
                f"epoch: [{epoch}], "
                f"step: [{epoch_step}/{len(self.train_loader)}], "
                f"batch loss: {loss:.5f}, "
                f"step time: {step_time:.5f}"
        )

    def log_metrics(self, epoch, loss, step_time):
        self.summary_writer.add_scalar("epoch", epoch, self.step)

        self.summary_writer.add_scalars(
                "loss",
                {"train": float(loss.item())},
                self.step
        )
        self.summary_writer.add_scalar(
                "time/data", step_time, self.step
        )

    def validate(self):
        results = {"preds": [], "gts": []}
        total_loss = 0
        self.model.eval()

        # No need to track gradients for validation, we're not optimizing.
        with no_grad():
            for batch, gts in self.val_loader:
                batch = batch.to(self.device)
                gts = gts.to(self.device)
                output = self.model(batch)
                loss = self.criterion(output, gts)
                total_loss += loss.item()
                preds = output.cpu().numpy()
                results["preds"].extend(list(preds))
                results["gts"].extend(list(gts.cpu().numpy()))


        average_loss = total_loss / len(self.val_loader)

        self.summary_writer.add_scalars(
                "loss",
                {"test": average_loss},
                self.step
        )
        print(f"validation loss: {average_loss:.5f}")

def CCLoss(x,y):
    vx = x - mean(x)
    vy = y - mean(y)
    loss = tsum(vx * vy) / (sqrt(tsum(vx ** 2)) * sqrt(tsum(vy ** 2)))
    return -loss


    


