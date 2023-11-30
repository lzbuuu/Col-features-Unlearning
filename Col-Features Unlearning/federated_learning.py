#!/usr/bin/python3

"""federated.py Contains an implementation of federated learning with ten
                workers applied to the Fashion MNIST data set for image
                classification using a slightly modified version of the LeNet5
                CNN architecture.

For the ID2223 Scalable Machine Learning course at KTH Royal Institute of
Technology"""

__author__ = "Xenia Ioannidou and Bas Straathof"


import os
import random
import time

import pandas as pd
import torch
import numpy as np
from utils import getDataset, splitData, map_splitLabels
from entity import Client, Server
import copy

# The Pysyft Federated Learning library


class FederateLearning:
    def __init__(self, args):
        self.args = args
        self.epochs = args.epochs
        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.num_clients = args.num_clients
        self.unlearned_id = args.unlearned_id
        self.clients, self.server = self.init()
        self.n_train_batches = self.clients[0].n_train_batches
        self.n_test_batches = self.clients[0].n_test_batches


    def init(self):
        train_set, test_set = getDataset(self.args)
        if self.dataset in ['BCW', 'Criteo', 'Adult', 'Diabetes']:
            n_total_features = train_set.shape[-1] - 1
            n = train_set.shape[0]
            idx = np.random.choice(range(n), int(n * 0.8))
            # if self.dataset == 'BCW':
            #     index = 0
            # else:
            #     index = -1
            # for i in list(set(range(n)).difference(set(list(idx)))):
            #     if train_set.iloc[i, index] == 0:
            #         train_set.iloc[i, index] = 1
            #     else:
            #         train_set.iloc[i, index] = 0
            #
            # for i in range(3):
            #     s = pd.Series(np.random.normal(0, 1, train_set.shape[0]))
            #     s[idx] = 0
            #     train_set.insert(n_total_features, 'Noises{}'.format(i), s)
            #     test_set.insert(n_total_features, 'Noises{}'.format(i), 0)
            #
            #     # train_set.insert(n_total_features, 'Noises{}'.format(i), i+1)
            #     # test_set.insert(n_total_features, 'Noises{}'.format(i), i+1)
            #     n_total_features += 1
        else:
            n_total_features = train_set.data.shape[-1]
        num_clients = self.num_clients
        clients = []
        pre_client_features = int(n_total_features / num_clients)
        start = 0
        for i in range(self.num_clients):
            if i == self.num_clients - 1:
                features_ids = list(range(start, n_total_features))
            else:
                features_ids = list(range(start, start + pre_client_features))
            if self.dataset in ['Criteo', 'BCW', 'Adult', 'Diabetes']:
                if i == self.num_clients+self.unlearned_id:
                    unlearning = True
                    if self.args.retain:
                        features_ids = features_ids[:-1]
                else:
                    unlearning = False
                if len(features_ids) != 0:
                    clients.append(Client(self.args, train_set, test_set, features_ids, unlearning))
                else:
                    self.num_clients -= 1
                    self.args.num_clients -= 1
            start += pre_client_features
        if self.dataset in ['Criteo', 'BCW', 'Adult', 'Diabetes']:
            server = Server(self.args, train_set, test_set)
        else:
            server = None
            exit()

        return clients, server

    def federated_train(self, save=True):
        start = time.perf_counter()
        n_train_batches, n_test_batches = self.n_train_batches, self.n_test_batches
        batch_size = self.batch_size
        for epoch_id in range(self.epochs):
            loss = 0.0
            total_n_train, total_n_test = 0, 0
            correct_train, correct_test = 0, 0
            print('Start train epoch {}'.format(epoch_id+1))
            for batch_id in range(n_train_batches):
                client_outputs_tensor = []
                for client in self.clients:
                    output_tensor = client.forward_round_per_batch(batch_id)
                    output_tensor.requires_grad_()
                    client_outputs_tensor.append(output_tensor)
                server_model_loss, client_outputs_tensor_grad, correct_pre_batch = self.server \
                    .forward_and_backward_round_per_batch(batch_id, client_outputs_tensor)
                loss += server_model_loss
                correct_train += correct_pre_batch
                total_n_train += batch_size
                for i, client in enumerate(self.clients):
                    client.backward_round_per_batch(client_outputs_tensor_grad[i])
                if batch_id % 50 == 0:
                    print('Train loss: {}'.format(server_model_loss.item()))
            accuracy = (correct_train / total_n_train) * 100
            print('Train {} of epoch {} - Accuracy: [{}/{} ({:.2f}%)]; Loss: {}'.format(self.dataset, epoch_id + 1,
                                                                                        correct_train, total_n_train,
                                                                                        accuracy,
                                                                                        loss.item()))
            for client in self.clients:
                client.scheduler_step()
            self.server.scheduler_step()
            # test a round(epoch)
            self.predict(epoch_id)
        end = time.perf_counter()
        correct_test, total_n_test, accuracy, loss = self.predict()
        print('-' * 50 + '\n')
        print('Complete the training process with accuracy [{}/{} ({:.2f}%)], running {}ms.'.format(correct_test,
                                                                                                    total_n_test,
                                                                                                    accuracy,
                                                                                                    (end-start)*1000))
        for i, client in enumerate(self.clients):
            client.original_model = copy.deepcopy(client.model)
        self.server.original_model = copy.deepcopy(self.server.model)
        if self.args.save:
            if self.args.retain:
                name = 'Retain'
            else:
                name = 'Original'
            path = 'D://result/{}/{}/'.format(self.args.dataset, name)
            if not os.path.exists(path):
                os.makedirs(path)
            for i, client in enumerate(self.clients):
                torch.save(client.model.state_dict(), '{}/client{}_model.state_dict'.format(path, i))
            torch.save(self.server.model.state_dict(), '{}/server_model.state_dict'.format(path))
        grad = 0
        for batch_id in range(n_train_batches):
            client_outputs_tensor = []
            for i, client in enumerate(self.clients):
                output_tensor = client.forward_round_per_batch(batch_id)
                output_tensor.requires_grad_()
                client_outputs_tensor.append(output_tensor)
            server_grad, client_outputs_tensor_grad = self.server.compute_grad(batch_id, client_outputs_tensor)
            grads = torch.nn.utils.parameters_to_vector(server_grad)
            for i, client in enumerate(self.clients):
                client_grad = client.compute_grad(client_outputs_tensor_grad[i])
                client_grad = torch.nn.utils.parameters_to_vector(client_grad)
                # grads = torch.cat((grads, client_grad), dim=0)
            grad = torch.sqrt((grads ** 2).sum()) / 2
        print('grad: {}'.format(grad))
        return (end - start) * 1000

    def finetuning(self,  unlearned_index=[0,1,2], unlearn_idx=-1):
        start = time.perf_counter()

        n_train_batches, n_test_batches = self.n_train_batches, self.n_test_batches
        batch_size = self.batch_size
        if unlearn_idx == -1:
            unlearn_idx = len(self.clients)-1
        unlearning_client = self.clients[unlearn_idx]
        unlearning_client.unlearn_features(unlearned_index)
        print('Finetuning client {} feature {}'.format(unlearn_idx, unlearned_index))
        self.federated_train(save=False)
        end = time.perf_counter()
        print('Complete the Finetuning process with running {}ms.'.format((end - start) * 1000))
        path = 'D://result/{}/{}'.format(self.args.dataset, 'Finetuning'+str(unlearned_index))
        if not os.path.exists(path):
            os.makedirs(path)
        for i, client in enumerate(self.clients):
            torch.save(client.model.state_dict(), '{}/client{}_model.state_dict'.format(path, i))
        torch.save(self.server.model.state_dict(), '{}/server_model.state_dict'.format(path))
        grad = 0
        for batch_id in range(n_train_batches):
            client_outputs_tensor = []
            for i, client in enumerate(self.clients):
                output_tensor = client.forward_round_per_batch(batch_id)
                output_tensor.requires_grad_()
                client_outputs_tensor.append(output_tensor)
            server_grad, client_outputs_tensor_grad = self.server.compute_grad(batch_id, client_outputs_tensor)
            grads = torch.nn.utils.parameters_to_vector(server_grad)
            for i, client in enumerate(self.clients):
                client_grad = client.compute_grad(client_outputs_tensor_grad[i])
                client_grad = torch.nn.utils.parameters_to_vector(client_grad)
                # grads = torch.cat((grads, client_grad), dim=0)
            grad = torch.sqrt((grads ** 2).sum()) / 2
        print('grad: {}'.format(grad))


    def predict(self, epoch_id=None):
        total_n_train, total_n_test = 0, 0
        correct_train, correct_test = 0, 0
        loss = []
        batches = self.n_test_batches
        for batch_idx in range(batches):
            client_test_outputs_tensor = []
            for client in self.clients:
                test_output_tensor = client.test_per_batch(batch_idx)
                client_test_outputs_tensor.append(test_output_tensor)
            correct_test_pre_batch, batch_loss = self.server.test_per_batch(batch_idx, client_test_outputs_tensor)
            correct_test += correct_test_pre_batch
            total_n_test += self.batch_size
            loss.append(batch_loss)
        accuracy = (correct_test / total_n_test) * 100
        if epoch_id is not None:
            print('Test {} of epoch {} - Accuracy: [{}/{} ({:.2f}%)]'.format(self.dataset, epoch_id + 1, correct_test,
                                                                             total_n_test,
                                                                             (correct_test / total_n_test) * 100))
        return correct_test, total_n_test, accuracy, loss

    def vertical_unlearning(self, unlearned_index=[0,1,2], unlearn_idx=-1):
        losses = []
        start = time.perf_counter()
        if unlearn_idx == -1:
            unlearn_idx = len(self.clients) - 1
        unlearning_client = self.clients[unlearn_idx]
        unlearning_client.unlearn_features(unlearned_index)
        print('Unlearning client {} feature {}'.format(unlearn_idx, unlearned_index))
        n_train_batches, n_test_batches = self.n_train_batches, self.n_test_batches
        # path = 'result/{}/{}'.format(self.args.dataset, 'Retain'+str(unlearned_index))
        # for i, client in enumerate(self.clients):
        #     client.model.load_state_dict(torch.load('{}/client{}_model.state_dict'.format(path, i)))
        # self.server.model.load_state_dict(torch.load('{}/server_model.state_dict'.format(path)))
        # grad = 0
        # for batch_id in range(n_train_batches):
        #     client_outputs_tensor = []
        #     for i, client in enumerate(self.clients):
        #         output_tensor = client.forward_round_per_batch(batch_id)
        #         output_tensor.requires_grad_()
        #         client_outputs_tensor.append(output_tensor)
        #     server_grad, client_outputs_tensor_grad = self.server.compute_grad(batch_id, client_outputs_tensor)
        #     grads = torch.nn.utils.parameters_to_vector(server_grad)
        #     for i, client in enumerate(self.clients):
        #         client_grad = client.compute_grad(client_outputs_tensor_grad[i])
        #         client_grad = torch.nn.utils.parameters_to_vector(client_grad)
        #         grads = torch.cat((grads, client_grad), dim=0)
        #     grad += torch.sqrt((grads**2).sum()/2)
        # print('grad: {}'.format(grad))
        client_outputs_tensor_grads = []
        for batch_id in range(n_train_batches):
            client_outputs_tensor = []
            for i, client in enumerate(self.clients):
                if i == unlearn_idx:
                    output_tensor = client.unlearn_forward_round_per_batch(batch_id)
                else:
                    output_tensor = client.forward_round_per_batch(batch_id)
                output_tensor.requires_grad_()
                client_outputs_tensor.append(output_tensor)
                client_outputs_tensor_grads.append(torch.zeros_like(output_tensor))
            client_outputs_tensor_grad = self.server.unlearn_backward_round_per_batch(batch_id, client_outputs_tensor)
            for i, client in enumerate(self.clients):
                if i == unlearn_idx:
                    grad = client_outputs_tensor_grad[i]
                    # client.unlearn_backward_round_per_batch(client_outputs_tensor_grad[i])
                else:
                    grad = client_outputs_tensor_grad[i] + self.server.inputs_tensor_grad[batch_id][i]
                    # client.backward_round_per_batch(
                    #     client_outputs_tensor_grad[i] + self.server.inputs_tensor_grad[batch_id][i])
                    # if batch_id == n_train_batches-1:
                    #     client.backward_round_per_batch(client_outputs_tensor_grad[i]+self.server.inputs_tensor_grad[batch_id][i])
                    # else:
                    #     client.backward_round_per_batch(client_outputs_tensor_grad[i]*0.0)
                client.unlearn_backward_round_per_batch(grad)
        # for i, client in enumerate(self.clients):
        #     if i == unlearn_idx:
        #         client.unlearn_update(tau=self.args.learning_rate)
        #     else:
        #         client.unlearn_update_1(tau=self.args.learning_rate)
        self.server.unlearn_update(tau=self.args.learning_rate)
        end = time.perf_counter()

        path = 'D://result/{}/{}/'.format(self.args.dataset, 'Unlearned'+str(unlearned_index))

        if not os.path.exists(path):
            os.makedirs(path)
        for i, client in enumerate(self.clients):
            torch.save(client.model.state_dict(), '{}client{}_model.state_dict'.format(path, i))
        torch.save(self.server.model.state_dict(), '{}server_model.state_dict'.format(path))

        correct_test, total_n_test, accuracy, loss = self.predict()
        # losses[-1] += loss
        print('-' * 50 + '\n')
        print('Complete the unlearn process with accuracy [{}/{} ({:.2f}%)], running {}ms.'.format(correct_test,
                                                                                                   total_n_test,
                                                                                                   accuracy,
                                                                                                   (
                                                                                                               end - start) * 1000))
        grad = 0
        for batch_id in range(n_train_batches):
            client_outputs_tensor = []
            for i, client in enumerate(self.clients):
                output_tensor = client.forward_round_per_batch(batch_id)
                output_tensor.requires_grad_()
                client_outputs_tensor.append(output_tensor)
            server_grad, client_outputs_tensor_grad = self.server.compute_grad(batch_id, client_outputs_tensor)
            grads = torch.nn.utils.parameters_to_vector(server_grad)
            for i, client in enumerate(self.clients):
                client_grad = client.compute_grad(client_outputs_tensor_grad[i])
                client_grad = torch.nn.utils.parameters_to_vector(client_grad)
                # grads = torch.cat((grads, client_grad), dim=0)
            grad = torch.sqrt((grads**2).sum())/2
        print('grad: {}'.format(grad))

        # losses = np.array(losses)
        # losses = pd.DataFrame(losses, index=None, columns=list(range(losses.shape[-1])))
        # losses.to_csv('C:/Users/MI少年/Desktop/1.csv')


    def retain(self):
        for i, client in enumerate(self.clients):
            client.retain()
        self.server.retain()
        start = time.perf_counter()
        runtime = self.federated_train()
        end = time.perf_counter()
        correct_test, total_n_test, accuracy, loss = self.predict()
        print('-' * 50 + '\n')
        print('Complete the retraining process with accuracy [{}/{} ({:.2f}%)], running {}ms.'.format(correct_test,
                                                                                                      total_n_test,
                                                                                                      accuracy,
                                                                                                      runtime))


def FederatedUnlearningFeatures(args, clients, server):
    unlearn_idx = args.unlearned_id
    unlearn_size = args.n_unlearned_features
    unlearned_client = clients[unlearn_idx]
    for epoch_idx in range(args.unlearn_epochs):
        for batch_idx in range(unlearned_client.n_train_batches):
            unlearned_client.unlearning_per_batch(batch_idx)

        total_n_train, total_n_test = 0, 0
        correct_train, correct_test = 0, 0
        batch_size = unlearned_client.batch_size
        for batch_idx in range(unlearned_client.n_test_batches):
            client_test_outputs_tensor = []
            for client in clients:
                test_output_tensor = client.test_per_batch(batch_idx)
                client_test_outputs_tensor.append(test_output_tensor)
            correct_test_pre_batch = server.test_per_batch(batch_idx, client_test_outputs_tensor)
            correct_test += correct_test_pre_batch
            total_n_test += batch_size
        print('Test {} of epoch {} - Accuracy: [{}/{} ({:.2f}%)]'
              .format(args.dataset, epoch_idx + 1, correct_test, total_n_test, (correct_test / total_n_test) * 100))
        if epoch_idx == args.unlearn_epochs-1:
            print('Complete the unlearning process with accuracy [{}/{} ({:.2f}%)]'.format(correct_test, total_n_test, (
                    correct_test / total_n_test) * 100))
    return unlearned_client, unlearn_idx


def FederatedUnlearningSamples(args, clients, server):
    unlearn_idx = args.unlearned_id
    unlearned_client = clients[unlearn_idx]
    unlearned_client.unlearn_samples()
    for client in clients:
        client.delete_samples()
    server.delete_labels()
    n_train_batches, n_test_batches = clients[0].n_train_batches, clients[0].n_test_batches
    batch_size = clients[0].batch_size
    for epoch_idx in range(1):
        val_loss, val_losses, val_accs = 0.0, [], []
        total_n_train, total_n_test = 0, 0
        correct_train, correct_test = 0, 0
        train_loss, test_loss = 0, 0
        # train a round
        print('Start unlearning train epoch {}'.format(epoch_idx))
        for batch_idx in range(n_train_batches):
            client_outputs_tensor = []
            for client in clients:
                output_tensor = client.forward_round_per_batch(batch_idx)
                output_tensor.requires_grad_()
                client_outputs_tensor.append(output_tensor)
            server_model_loss, client_outputs_tensor_grad, correct_pre_batch = server \
                .forward_and_backward_round_per_batch(batch_idx, client_outputs_tensor)
            correct_train += correct_pre_batch
            total_n_train += batch_size
            for i in range(len(clients)):
                clients[i].backward_round_per_batch(client_outputs_tensor_grad[i])
            print('Accuracy: [{}/{} ({:.2f}%)]'.format(correct_pre_batch, batch_size,
                                                       (correct_pre_batch / batch_size) * 100))
            if (batch_idx + 1) % 10 == 0:
                print('Train loss of batch {} in epoch {} : {:.6f}\n'.format(batch_idx + 1, epoch_idx,
                                                                             server_model_loss.item()))
            # del client_outputs_tensor, server_model_loss, client_outputs_tensor_grad
        print('Train {} of epoch {} - Accuracy: [{}/{} ({:.2f}%)]'
              .format(args.dataset, epoch_idx + 1, correct_train, total_n_train, (correct_train / total_n_train) * 100))

        # test a round
        total_n_train, total_n_test = 0, 0
        correct_train, correct_test = 0, 0
        for batch_idx in range(n_test_batches):
            client_test_outputs_tensor = []
            for client in clients:
                test_output_tensor = client.test_per_batch(batch_idx)
                client_test_outputs_tensor.append(test_output_tensor)
            correct_test_pre_batch = server.test_per_batch(batch_idx, client_test_outputs_tensor)
            correct_test += correct_test_pre_batch
            total_n_test += batch_size
        print('Test {} of epoch {} - Accuracy: [{}/{} ({:.2f}%)]'
              .format(args.dataset, epoch_idx + 1, correct_test, total_n_test, (correct_test / total_n_test) * 100))
        if epoch_idx == args.epochs - 1:
            print('Complete the training process with accuracy [{}/{} ({:.2f}%)]'.format(correct_test, total_n_test, (
                        correct_test / total_n_test) * 100))
    accuracy = TestBackdoor(clients, server)
    print('Backdoor accuracy after unlearning: {:.2f}%'.format(accuracy * 100))


def TestBackdoor(clients, server):
    # total_n = len(clients[0].poison_test_indices)
    batches = int(len(clients[0].poison_test_indices)/clients[0].batch_size)
    total_n = clients[0].batch_size*batches
    total_correct_n = 0
    for batch_idx in range(batches):
        client_test_outputs_tensor = []
        for client in clients:
            output_tensor = client.test_per_batch(batch_idx, True)
            client_test_outputs_tensor.append(output_tensor)
        correct_test_pre_batch = server.test_per_batch(batch_idx, client_test_outputs_tensor, True)
        total_correct_n += correct_test_pre_batch
    return total_correct_n/total_n
