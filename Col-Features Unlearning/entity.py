import copy
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from models import BottomModel, TopModel
from collections import OrderedDict
import utils


class Client:
    def __init__(self, args, train_set, test_set, client_features_ids, unlearning=False):
        self.args = args
        self.backdoor = unlearning
        self.client_features_ids = client_features_ids
        self.train_loader, self.test_loader = utils.getDataLoader(args, train_set, test_set, client_features_ids)
        # self.train_test_loader, self.test_test_loader = utils.getDataLoader(args, train_test, test_test, client_features_ids)
        self.batch_size = self.train_loader.batch_size
        self.epochs = args.epochs
        self.n_train_batches = len(self.train_loader)
        self.n_test_batches = len(self.test_loader)
        # self.n_train_test_batches = len(self.train_test_loader)
        # self.n_test_test_batches = len(self.test_test_loader)
        self.output_tensor_grad = torch.Tensor([]).to(args.device)
        self.output_tensor = torch.Tensor([]).to(args.device)
        self.loss_crit = nn.CrossEntropyLoss(reduction='sum')
        self.model = BottomModel(args.dataset).get_model(len(client_features_ids), args.num_clients).to(args.device)
        self.original_model = self.unlearned_model = self.retain_model = None
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        self.original_grad = []
        for parm in self.model.parameters():
            self.original_grad.append(torch.zeros_like(parm.data, device=args.device))
        self.last_batch = False
        self.current_grad = copy.deepcopy(self.original_grad)

    def forward_round_per_batch(self, batch_id):
        if batch_id == 0:
            for i, parm in enumerate(self.model.parameters()):
                self.original_grad[i] *= 0.0
        elif batch_id == self.n_train_batches-1:
            self.last_batch = True
        data, _ = self.train_loader[batch_id]
        data = data.to(self.args.device)
        self.model.train()
        self.output_tensor = self.model(data)
        return self.output_tensor.data

    def unlearn_forward_round_per_batch(self, batch_id):
        data, _ = self.train_loader[batch_id]
        data = data.to(self.args.device)
        self.model.train()
        self.output_tensor = self.model(data)
        return self.output_tensor.data

    def backward_round_per_batch(self, client_outputs_tensor_grad):
        self.output_tensor_grad.data = client_outputs_tensor_grad.data
        self.optimizer.zero_grad()
        self.output_tensor.backward(client_outputs_tensor_grad)
        for i, parm in enumerate(self.model.parameters()):
            self.original_grad[i] += parm.grad.clone()
        if self.last_batch:
            for i in range(len(self.original_grad)):
                self.original_grad[i] /= self.n_train_batches
            self.last_batch = False
        self.optimizer.step()

    def compute_grad(self, client_outputs_tensor_grad):
        self.output_tensor_grad.data = client_outputs_tensor_grad.data
        self.optimizer.zero_grad()
        self.output_tensor.backward(client_outputs_tensor_grad)
        grads = []
        for i, parm in enumerate(self.model.parameters()):
            grads.append(parm.grad)
        return grads

    def unlearn_backward_round_per_batch(self, client_outputs_tensor_grad):
        self.output_tensor_grad.data = client_outputs_tensor_grad.data
        self.output_tensor.backward(client_outputs_tensor_grad)
        for i, parm in enumerate(self.model.parameters()):
            self.current_grad[i] += parm.grad.clone()

    def unlearn_update(self, tau=0.01):
        for i, parm in enumerate(self.model.parameters()):
            parm.data = parm.data+tau*(self.current_grad[i]/self.n_train_batches-self.original_grad[i])

    def unlearn_update_1(self, tau=0.01):
        for i, parm in enumerate(self.model.parameters()):
            parm.data = parm.data-tau*(self.current_grad[i]/self.n_train_batches)

    def scheduler_step(self):
        self.scheduler.step()

    def retain(self):
        args = self.args
        self.output_tensor_grad = torch.Tensor([]).to(args.device)
        self.output_tensor = torch.Tensor([]).to(args.device)
        self.loss_crit = nn.CrossEntropyLoss(reduction='sum')
        self.model = BottomModel(args.dataset).get_model(len(self.client_features_ids), args.num_clients).to(
            args.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        self.original_grad = []
        for parm in self.model.parameters():
            self.original_grad.append(torch.zeros_like(parm.data, device=args.device))
        self.last_batch = False
        self.current_grad = copy.deepcopy(self.original_grad)

    def test_per_batch(self, batch_id):
        data, _ = self.test_loader[batch_id]
        data = data.to(self.args.device)
        self.model.eval()
        test_output_tensor = self.model(data)
        return test_output_tensor

    def test_per_batch1(self, batch_id):
        data, _ = self.train_test_loader[batch_id]
        data = data.to(self.args.device)
        self.model.eval()
        test_output_tensor = self.model(data)
        return test_output_tensor

    def unlearn_features(self, unlearned_index):
        if len(unlearned_index) <= 0:
            unlearned_index = list(range(len(self.client_features_ids)))
        for idx in unlearned_index:
            self.train_loader.data[:, idx] *= 0.
            self.test_loader.data[:, idx] *= 0.


class Server:
    def __init__(self, args, train_set, test_set,):
        self.model = TopModel(args.dataset).get_model(args.num_clients).to(args.device)
        self.original_model = self.unlearned_model = self.retain_model = None
        self.args = args
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.train_labels, self.test_labels = utils.getDataLabels(args, train_set, test_set)
        # self.train_test, self.test_test = utils.getDataLabels(args, train_test, test_test)
        self.n_train_batches = int(len(self.train_labels) / self.batch_size)+1
        self.n_test_batches = int(len(self.test_labels) / self.batch_size)+1
        # self.n_train_test_batches = int(len(self.train_test) / self.batch_size) + 1
        # self.n_test__test_batches = int(len(self.test_test) / self.batch_size) + 1
        self.loss_crit = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        self.original_grad = []
        self.inputs_tensor_grad = []
        for parm in self.model.parameters():
            self.original_grad.append(torch.zeros_like(parm.data, device=args.device))

        self.current_grad = copy.deepcopy(self.original_grad)

    def forward_and_backward_round_per_batch(self, batch_idx, inputs_tensor):
        data = torch.cat(inputs_tensor, dim=1)
        if batch_idx == 0:
            for i in range(len(self.original_grad)):
                self.original_grad[i] *= 0.0
            self.inputs_tensor_grad = []
            # for i in range(len(inputs_tensor)):
            #     self.inputs_tensor_grad.append(torch.zeros_like(inputs_tensor[i]))
        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        if stop > len(self.train_labels):
            stop = -1
        self.model.train()
        output_tensor = self.model(data)
        batch_target = torch.tensor(self.train_labels[start:stop], dtype=torch.long).to(self.args.device)
        loss = self.loss_crit(output_tensor, batch_target)
        self.optimizer.zero_grad()
        loss.backward()
        for i, parm in enumerate(self.model.parameters()):
            self.original_grad[i] += parm.grad.clone()
        self.optimizer.step()
        inputs_tensor_grad = [i.grad for i in inputs_tensor]
        self.inputs_tensor_grad.append(inputs_tensor_grad)
        if batch_idx == self.n_train_batches-1:
            for i in self.original_grad:
                i /= self.n_train_batches
        _, predictions = output_tensor.max(1)
        correct_per_batch = predictions.eq(batch_target).sum().item()
        return loss, inputs_tensor_grad, correct_per_batch

    def unlearn_backward_round_per_batch(self, batch_idx, inputs_tensor):
        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        if stop > len(self.train_labels):
            stop = -1
        data = torch.cat(inputs_tensor, dim=1)
        self.model.train()
        output_tensor = self.model(data)
        batch_target = torch.tensor(self.train_labels[start:stop], dtype=torch.long).to(self.args.device)
        loss = self.loss_crit(output_tensor, batch_target)
        loss.backward()
        for i, parm in enumerate(self.model.parameters()):
            self.current_grad[i] += parm.grad.clone()
        if batch_idx == self.n_train_batches-1:
            for i in range(len(self.current_grad)):
                self.current_grad[i] /= self.n_train_batches
        inputs_tensor_grad = [i.grad for i in inputs_tensor]
        return inputs_tensor_grad

    def compute_grad(self, batch_idx, inputs_tensor):
        data = torch.cat(inputs_tensor, dim=1)
        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        if stop > len(self.train_labels):
            stop = -1
        self.model.train()
        output_tensor = self.model(data)
        batch_target = torch.tensor(self.train_labels[start:stop], dtype=torch.long).to(self.args.device)
        loss = self.loss_crit(output_tensor, batch_target)
        self.optimizer.zero_grad()
        loss.backward()
        grads = []
        for parm in self.model.parameters():
            grads.append(parm.grad*len(inputs_tensor))
        inputs_tensor_grad = [i.grad for i in inputs_tensor]
        return grads, inputs_tensor_grad

    def unlearn_update(self, tau=0.01):
        for i, parm in enumerate(self.model.parameters()):
            parm.data = parm.data + tau * (self.current_grad[i] - self.original_grad[i])

    def test_per_batch(self, batch_idx, inputs_tensor, backdoor=False):
        data = inputs_tensor[0]
        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        if stop > len(self.test_labels):
            stop = -1
        for tensor in inputs_tensor[1:]:
            data = torch.cat((data, tensor), dim=1)
        self.model.eval()
        true_labels = torch.tensor(self.test_labels[start:stop], dtype=torch.long).to(self.args.device)
        test_output_tensor = self.model(data)
        loss = self.loss_crit(test_output_tensor, true_labels)
        _, predictions = test_output_tensor.max(1)
        correct_test_per_batch = predictions.eq(true_labels).sum().item()
        return correct_test_per_batch, loss.item()

    def test_per_batch1(self, batch_idx, inputs_tensor, backdoor=False):
        data = inputs_tensor[0]
        start = batch_idx * self.batch_size
        stop = start + self.batch_size
        if stop > len(self.train_test):
            stop = -1
        for tensor in inputs_tensor[1:]:
            data = torch.cat((data, tensor), dim=1)
        self.model.eval()
        true_labels = torch.tensor(self.train_test[start:stop], dtype=torch.long).to(self.args.device)
        test_output_tensor = self.model(data)
        loss = self.loss_crit(test_output_tensor, true_labels)
        _, predictions = test_output_tensor.max(1)
        correct_test_per_batch = predictions.eq(true_labels).sum().item()
        return correct_test_per_batch, loss.item()

    def scheduler_step(self):
        self.scheduler.step()

    def retain(self):
        args = self.args
        self.model = TopModel(args.dataset).get_model(args.num_clients).to(args.device)
        self.loss_crit = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=args.learning_rate, momentum=args.momentum,
                                         weight_decay=args.weight_decay)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.epochs)
        self.original_grad = []
        self.inputs_tensor_grad = []
        for parm in self.model.parameters():
            self.original_grad.append(torch.zeros_like(parm.data, device=args.device))
        self.current_grad = copy.deepcopy(self.original_grad)
