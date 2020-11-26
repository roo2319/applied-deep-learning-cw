import time
from typing import Union
from numpy.core.fromnumeric import squeeze
from torch import nn, optim, Tensor, no_grad, save, reshape, cat, unsqueeze, squeeze
import torch
from torch.nn import functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from torch import device as Device
import numpy as np
from .dataset import Salicon

class AdversarialTrainer:
    def __init__(
        self,
        generator: nn.Module,
        discriminator: nn.Module,
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
        self.batch_size = batch_size
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.dis_criterion = nn.BCELoss()
        self.optimizer = SGD(self.generator.parameters(),lr=0.03, momentum=0.9, weight_decay=0.0005, nesterov=True) 
        self.dis_optimizer = SGD(self.discriminator.parameters(),lr=0.03, momentum=0.9,weight_decay=0.0005,nesterov=True)
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
            self.generator.train()
            self.discriminator.train()
            for batch, gts in self.train_loader:
                # LR decay
                # need to update learning rate between 0.03 and 0.0001 (according to paper)
            # -------------------------- GENERATOR -----------------------------
                
                optimstate = self.optimizer.state_dict()
                self.optimizer = SGD(self.generator.parameters(),lr=lrs[epoch], momentum=0.9, weight_decay=0.0005, nesterov=True)
                self.optimizer.load_state_dict(optimstate)

                self.optimizer.zero_grad()
                # load batch to device
                batch = batch.to(self.device)
                gts = gts.to(self.device)

                # train step
                step_start_time = time.time()
                output = self.generator.forward(batch)

                loss = self.criterion(output,gts)
                loss.backward()
                self.optimizer.step()


            # -------------------- DISCRIMINATOR --------------------------------


                self.dis_optimizer.zero_grad()
                scaled_image = F.interpolate(batch,(48,48))
                out_images = unsqueeze(reshape(output,(self.batch_size,48,48)),1)
                gt_images = unsqueeze(reshape(gts,(self.batch_size,48,48)),1)
                
                pred_stack = cat((scaled_image,
                                  out_images), dim=1)
                gt_stack = cat((scaled_image,
                                  gt_images), dim=1)

                stack = cat((pred_stack,gt_stack),dim=0)
                r = torch.randperm(self.batch_size * 2)
                labels = cat((torch.zeros(self.batch_size),torch.ones(self.batch_size))) 
                stack = stack[r][:]
                labels = labels[r]
                labels = labels.to(self.device)

                dis_loss = self.discriminator.forward(stack)
                # print(dis_loss)
                
                dis_loss = squeeze(dis_loss)
                dis_loss = self.dis_criterion(dis_loss, labels)
                dis_loss.backward()
                self.dis_optimizer.step()

            #  --------------------------- LOGS --------------------------------

                if ((self.step + 1) % log_frequency) == 0:
                    with no_grad():
                        accuracy = compute_accuracy(gts, output)
                    step_time = time.time() - step_start_time
                    self.log_metrics(epoch, accuracy, loss, step_time)
                    self.print_metrics(epoch, accuracy, loss, step_time)

                # tilo says to use weight decay of 0.0005 (dunno what that means)
                # tilo also says dont worry about momentum decay (nice)

                # count steps
                self.step += 1

            # log epoch 
            self.summary_writer.add_scalar("epoch", epoch, self.step)

            # validate
            if ((epoch + 1) % val_frequency) == 0:
                self.validate()
                self.generator.train()
            if (epoch+1) % 10 == 0:
                save(self.generator,"checkp_model.pkl") 


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
        results = {"preds": [], "gts": []}
        total_loss = 0
        self.generator.eval()

        # No need to track gradients for validation, we're not optimizing.
        with no_grad():
            for batch, gts in self.val_loader:
                print(batch.shape,gts.shape)
                exit()
                batch = batch.to(self.device)
                gts = gts.to(self.device)
                output = self.generator(batch)
                loss = self.criterion(output, gts)
                total_loss += loss.item()
                preds = output.cpu().numpy()
                results["preds"].extend(list(preds))
                results["gts"].extend(list(gts.cpu().numpy()))

        accuracy = compute_accuracy(
            np.array(results["gts"]), np.array(results["preds"])
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
    gts: Union[Tensor, np.ndarray], preds: Union[Tensor, np.ndarray]
) -> float:
    """
    Args:
        gts: ``(batch_size, class_count)`` tensor or array containing example gts
        preds: ``(batch_size, class_count)`` tensor or array containing generator prediction
    """
   
    return 0


    


