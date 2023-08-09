from os import XATTR_SIZE_MAX
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional
import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import MNIST
from flwr.common import Metrics
import brevitas.nn as qnn
from concrete.ml.torch.compile import compile_brevitas_qat_model
from sklearn.metrics import log_loss, accuracy_score
from brevitas import config
from torch.nn.utils import clip_grad_norm_

config.IGNORE_MISSING_KEYS = True


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLIENTS = 5
N_FEAT = 784
n_bits = 3
LR = 1e-2
NOISE_MULTIPLIER = 0.25
MAX_GRAD_NORM = 1.2


def load_datasets():
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5)),
            transforms.Lambda(lambda x: torch.flatten(x)),
        ]
    )
    trainset = MNIST("./dataset", train=True, download=True, transform=transform)
    testset = MNIST("./dataset", train=False, download=True, transform=transform)

    partition_size = len(trainset) // NUM_CLIENTS
    lengths = [partition_size] * NUM_CLIENTS
    datasets = random_split(trainset, lengths, torch.Generator().manual_seed(42))

    trainloaders = []
    valloaders = []
    for ds in datasets:
        len_val = len(ds) // 10
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        trainloaders.append(DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True))
        valloaders.append(DataLoader(ds_val, batch_size=BATCH_SIZE))
    testloader = DataLoader(testset, batch_size=BATCH_SIZE)
    return trainloaders, valloaders, testloader


class Net(nn.Module):
    def __init__(self):
      super(Net, self).__init__()

      self.quant_inp = qnn.QuantIdentity(bit_width=n_bits, return_quant_tensor=True)
      self.fc1 = qnn.QuantLinear(N_FEAT, 100, True, weight_bit_width=n_bits, bias_quant=None)
      self.relu1 = qnn.QuantReLU(bit_width=n_bits, return_quant_tensor=True)
      self.fc2 = qnn.QuantLinear(100, 10, True, weight_bit_width=n_bits, bias_quant=None)

    def forward(self, x):
      x = self.quant_inp(x)
      x = self.relu1(self.fc1(x))
      x = self.fc2(x)
      return x


def train(net, trainloader, epochs: int, verbose=False):
    criterion = nn.CrossEntropyLoss()
    
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        optimizer = torch.optim.SGD(net.parameters(), lr=LR)
        
        for X_batch, y_batch in trainloader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            
            y_hat = net(X_batch)
            loss = criterion(y_hat, y_batch)
            loss.backward()

            for param in net.parameters():
                noise = torch.normal(mean=0, std=NOISE_MULTIPLIER * MAX_GRAD_NORM, size=param.grad.shape)
                param.grad += noise

            torch.nn.utils.clip_grad_norm_(net.parameters(), MAX_GRAD_NORM)

            optimizer.step()

            epoch_loss += loss.item()
            total += len(y_batch)
            correct += (torch.max(y_hat, 1)[1] == y_batch).sum().item()

        epoch_loss /= len(trainloader)
        epoch_acc = correct / total
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")


def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def get_parameters(net) -> List[np.ndarray]:
    params = [val.cpu().numpy() for k, val in net.state_dict().items() if "act_quant" not in k]
    return params


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=False)


class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=5)
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(accuracy)}
    

def client_fn(cid: str, trainloaders, valloaders ) -> FlowerClient:
    net = Net().to(DEVICE)

    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]

    return FlowerClient(net, trainloader, valloader)


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}


def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
    valloaders,
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:

    valloader = valloaders[0]

    X_test = next(iter(valloader))[0].numpy()
    y_test =  next(iter(valloader))[1].numpy()

    net = Net()
    set_parameters(net, parameters)

    quantized_module = compile_brevitas_qat_model(
        net,
        next(iter(valloader))[0].numpy(),
    )

    X_test_q = quantized_module.quantize_input(X_test)
    y_pred = quantized_module.quantized_forward(X_test_q, fhe="simulate")
    y_pred = quantized_module.dequantize_output(y_pred)

    loss = log_loss(y_test, y_pred, labels=[x for x in range(10)])
    accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))

    print(f"SERVER EVAL:\n\tLOSS: {loss}\n\tACCURACY:{accuracy}")

    return loss, {"accuracy": accuracy}


def main():
    trainloaders, valloaders, testloader = load_datasets()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.5,
        min_fit_clients=1,
        min_evaluate_clients=2,
        min_available_clients=5,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=lambda x, y, z: evaluate(x, y, z, valloaders),
    )

    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}

    fl.simulation.start_simulation(
        client_fn = lambda x: client_fn(x, trainloaders, valloaders),
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=10),
        strategy=strategy,
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()