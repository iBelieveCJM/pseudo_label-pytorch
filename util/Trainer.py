#!coding:utf-8
import torch
from torch.distributions.categorical import Categorical
from torch.nn import functional as F

from pathlib import Path
from util.datasets import NO_LABEL

class PseudoLabel:

    def __init__(self, model, optimizer, loss_fn, device, config, writer=None, save_dir=None, save_freq=5):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.device = device
        self.writer = writer
        self.labeled_bs = config.labeled_batch_size
        self.global_step = 0
        self.epoch = 0
        self.T1, self.T2 = config.t1, config.t2
        self.af = config.af
        
    def _iteration(self, data_loader, print_freq, is_train=True):
        loop_loss = []
        accuracy = []
        labeled_n = 0
        mode = "train" if is_train else "test"
        for batch_idx, (data, targets) in enumerate(data_loader):
            self.global_step += batch_idx
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)
            if is_train:
                labeled_bs = self.labeled_bs
                labeled_loss = torch.sum(self.loss_fn(outputs, targets)) / labeled_bs
                with torch.no_grad():
                    pseudo_labeled = outputs.max(1)[1]
                unlabeled_loss = torch.sum(targets.eq(NO_LABEL).float() * self.loss_fn(outputs, pseudo_labeled)) / (data.size(0)-labeled_bs +1e-10)
                loss = labeled_loss + self.unlabeled_weight()*unlabeled_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            else:
                labeled_bs = data.size(0)
                labeled_loss = unlabeled_loss = torch.Tensor([0])
                loss = torch.mean(self.loss_fn(outputs, targets))
            labeled_n += labeled_bs

            loop_loss.append(loss.item() / len(data_loader))
            acc = targets.eq(outputs.max(1)[1]).sum().item()
            accuracy.append(acc)
            if print_freq>0 and (batch_idx%print_freq)==0:
                print(f"[{mode}]loss[{batch_idx:<3}]\t labeled loss: {labeled_loss.item():.3f}\t unlabeled loss: {unlabeled_loss.item():.3f}\t loss: {loss.item():.3f}\t Acc: {acc/labeled_bs:.3%}")
            if self.writer:
                self.writer.add_scalar(mode+'_global_loss', loss.item(), self.global_step)
                self.writer.add_scalar(mode+'_global_accuracy', acc/labeled_bs, self.global_step)
        print(f">>>[{mode}]loss\t loss: {sum(loop_loss):.3f}\t Acc: {sum(accuracy)/labeled_n:.3%}")
        if self.writer:
            self.writer.add_scalar(mode+'_epoch_loss', sum(loop_loss), self.epoch)
            self.writer.add_scalar(mode+'_epoch_accuracy', sum(accuracy)/labeled_n, self.epoch)

        return loop_loss, accuracy

    def unlabeled_weight(self):
        alpha = 0.0
        if self.epoch > self.T1:
            alpha = (self.epoch-self.T1) / (self.T2-self.T1)*self.af
            if self.epoch > self.T2:
                alpha = af
        return alpha
        
    def train(self, data_loader, print_freq=20):
        self.model.train()
        with torch.enable_grad():
            loss, correct = self._iteration(data_loader, print_freq)

    def test(self, data_loader, print_freq=10):
        self.model.eval()
        with torch.no_grad():
            loss, correct = self._iteration(data_loader, print_freq, is_train=False)

    def loop(self, epochs, train_data, test_data, scheduler=None, print_freq=-1):
        for ep in range(epochs):
            self.epoch = ep
            if scheduler is not None:
                scheduler.step()
            print("------ Training epochs: {} ------".format(ep))
            self.train(train_data, print_freq)
            print("------ Testing epochs: {} ------".format(ep))
            self.test(test_data, print_freq)
            if ep % self.save_freq == 0:
                self.save(ep)

    def save(self, epoch, **kwargs):
        if self.save_dir is not None:
            model_out_path = Path(self.save_dir)
            state = {"epoch": epoch,
                    "weight": self.model.state_dict()}
            if not model_out_path.exists():
                model_out_path.mkdir()
            torch.save(state, model_out_path / "model_epoch_{}.pth".format(epoch))
