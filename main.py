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

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
NUM_CLIENTS = 5
N_FEAT = 784
n_bits = 3



from brevitas import config
config.IGNORE_MISSING_KEYS = True

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

    #print(trainset[0][0])

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


# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output

# class Net(nn.Module):
#   def __init__(self):
#     super(Net, self).__init__()

#     self.fc1 = nn.Linear(N_FEAT, 100)
#     self.fc2 = nn.Linear(100, 10)

#   def forward(self, x):
#     x = torch.relu(self.fc1(x))
#     x = self.fc2(x)
#     return x


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
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()
    for epoch in range(epochs):
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
        epoch_loss /= len(trainloader.dataset)
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
    #print(params[0])
    #print(f"KEYS IN NET: {net.state_dict().keys()}")
    return params


def set_parameters(net, parameters: List[np.ndarray]):
    params_dict = zip(net.state_dict().keys(), parameters)
    #print(params_dict)
    #params_dict = [(x,y) for (x,y) in zip(net.state_dict().keys(), parameters) ]
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    #print(f"STATE DICT KEYS: {list(state_dict.keys()).filter(lambda x, y: 'act_quant' not in x)}")
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
        train(self.net, self.trainloader, epochs=1)
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


# def evaluate(
#     server_round: int,
#     parameters: fl.common.NDArrays,
#     config: Dict[str, fl.common.Scalar],
#     valloaders,
# ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
#     net = Net().to(DEVICE)
#     valloader = valloaders[0]
#     set_parameters(net, parameters)
#     loss, accuracy = test(net, valloader)
#     print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
#     return loss, {"accuracy": accuracy}


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

    #y_pred = np.argmax(y_pred, axis=1)

    # def softmax(x):
    #   """Compute softmax values for each sets of scores in x."""
    #   e_x = np.exp(x - np.max(x))
    #   return e_x / e_x.sum()

    # y_pred = softmax(y_pred)

    # print(y_pred[0])
    # print(y_pred.shape)
    # print(y_test[0])
    # print(y_test.shape)

    #print(log_loss(y_test[0], y_pred))


    loss = log_loss(y_test, y_pred, labels=[x for x in range(10)])
    accuracy = accuracy_score(y_test, np.argmax(y_pred, axis=1))

    print(f"SERVER EVAL:\n\tLOSS: {loss}\n\tACCURACY:{accuracy}")

    return loss, {"accuracy": accuracy}
    # net = Net().to(DEVICE)
    # valloader = valloaders[0]
    # set_parameters(net, parameters)
    # loss, accuracy = test(net, valloader)
    # print(f"Server-side evaluation loss {loss} / accuracy {accuracy}")
    # return loss, {"accuracy": accuracy}

def main():
    trainloaders, valloaders, testloader = load_datasets()

    # trainloader = trainloaders[0]
    # valloader = valloaders[0]
    # net = Net().to(DEVICE)

    # print("="*10 + "Normal Net" + "="*10)

    # for epoch in range(0):
    #     train(net, trainloader, 1)
    #     loss, accuracy = test(net, valloader)
    #     print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    # loss, accuracy = test(net, testloader)
    # print(f"Final test set performance:\n\tloss {loss}\n\taccuracy {accuracy}")

    # print("="*10 + "Quantized Net" + "="*10)

    # fhe_net = Net()

    # for epoch in range(1):
    #   train(fhe_net, trainloader, 1)
    #   loss, accuracy = test(fhe_net, valloader)
    #   print(f"Epoch {epoch+1}: validation loss {loss}, accuracy {accuracy}")

    # print(fhe_net.state_dict().keys())

    # params = get_parameters(fhe_net)

    # #print(params.keys())

    # #sd = fhe_net.state_dict()

    # #print(params)

    # fhe_net = Net()

    # #fhe_net.load_state_dict(sd)

    # set_parameters(fhe_net, params)

    # quantized_module = compile_brevitas_qat_model(
    #     fhe_net,
    #     next(iter(trainloader))[0].numpy(),
    # )

    # y_true = next(iter(valloader))[1].numpy()

    # x_test_q = quantized_module.quantize_input(next(iter(valloader))[0].numpy())
    # y_pred = quantized_module.quantized_forward(x_test_q, fhe="simulate")
    # y_pred = quantized_module.dequantize_output(y_pred)

    # y_pred = np.argmax(y_pred, axis=1)

    # accuracy = np.sum(np.equal(y_pred, y_true))/len(y_true)

    # print(f"FHE Test accuracy: {accuracy*100:.2f}")

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
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
        client_resources=client_resources,
    )


if __name__ == "__main__":
    main()