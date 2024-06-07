# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import torch
import torch.nn as nn
import pennylane as qml
from torch.nn import Module
import math
    
# Hybrid QNN ####################################################################################################
class Q_linear(Module):
    in_features: int
    weight: torch.Tensor
    
    def __init__(self, in_features: int, n_layers:int, 
                 bias: bool = False, device=None, dtype=None) -> None:
        super().__init__()
        self.in_features = in_features
        self.n_layers = n_layers
        self.dev = qml.device("lightning.gpu", wires = in_features) if torch.cuda.is_available() else qml.device("default.qubit", wires = in_features)

        @qml.qnode(self.dev, interface="torch")
        def quantum_circuit(inputs, weights):
            #print(f"#################weights = {weights}#################")
            '''
            # Hadamard Layer
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            '''
            # Embedding layer
            qml.AngleEmbedding(inputs, wires=range(self.in_features))

            # Variational layer
            for _ in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights, wires=range(self.in_features))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.in_features)]
        
        weight_shapes = {"weights": (self.n_layers, self.in_features, 3)}
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes= weight_shapes)
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.qlayer(input)

### This is a class for using Quanvolution with any torch model
class Q_conv(Module):
    def __init__(self,  kernel_size: int, n_layers: int, stride: int, device=None, dtype=None)->None:
        super(Q_conv, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(self.kernel_size**2)
        self.n_layers = n_layers
        self.stride = stride
        
        # First define a q-node
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        #init_method = {"weights": self.weights}
        dev = qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        #dev = qml.device("default.qubit", wires = n_qubits)

        @qml.qnode(dev, interface="torch")
        def quantum_circuit(inputs, weights):
            #print(f"#################weights = {weights}#################")
            '''
            # Hadamard Layer # Increases complexity and time of training
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            '''
            # Embedding layer
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # Variational layer
            for _ in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
                
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

        #self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes, init_method)
        self.qlayer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Assert the rules
        assert len(input.shape) == 4                                                # (batch_size, n_channels, x_dim, y_dim)
        assert input.shape[-1] == input.shape[-2]
        assert self.stride > 0
        assert input.shape[1] == 1                                                  # Supports only one channel only

        k = self.kernel_size                                                             # kernel dimension
        output = None
        iterator = 0
        output_shape = int((input.shape[-1]-k)/self.stride + 1)
        #print(f"output_shape = {output_shape}\n")
        for i in range(0, input.shape[-2], self.stride):
            if(i+k > input.shape[-2]):
                break
            for j in range(0, input.shape[-1], self.stride):
                if(j+k > input.shape[-1]):
                    break
                x = torch.flatten(input[:,:, i:i+k, j:j+k], start_dim=1)
                exp_vals = self.qlayer(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output




  
class DressedQuantumNet(nn.Module):
    def __init__(self, input_shape, n_qubits=5, n_layers=3, n_op = 2):
        super().__init__()
        # The output from the penultimate layer of the classical models has 16 dimensions
        self.pre_net = nn.Linear(input_shape, n_qubits)
        self.qlayer = Q_linear(in_features=n_qubits, n_layers=n_layers)
        self.post_net = nn.Linear(n_qubits, n_op)

    def forward(self, x):
        x = torch.tanh(self.pre_net(x)) * torch.pi / 2.0
        x = torch.relu(self.qlayer(x))
        x = torch.log_softmax(self.post_net(x), dim=1)
        return x

#%% Full-fledged quanvolution
class QConv2D(nn.Module):
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, device=None, dtype=None)->None:
        super(QConv2D, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(in_channels * self.kernel_size**2)
        self.n_layers = n_layers
        self.stride = stride

        # First define a q-node
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        dev = qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        qnode = qml.QNode(self.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def quantum_circuit(self, inputs, weights):
            '''
            # Hadamard Layer # Increases complexity and time of training
            for wire in range(n_qubits):
                qml.Hadamard(wires = wire)
            '''
            # Embedding layer
            qml.AngleEmbedding(inputs, wires=range(self.n_qubits))

            # Variational layer
            for _ in range(self.n_layers):
                qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))

            return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Assert the rules
        assert len(input.shape) == 4                                                # (batch_size, n_channels, x_dim, y_dim)
        assert input.shape[-1] == input.shape[-2]
        assert self.stride > 0
        #assert input.shape[1] == 1                                                  # Supports only one channel only

        k = self.kernel_size                                                             # kernel dimension
        output = None
        iterator = 0
        output_shape = int((input.shape[-1]-k)/self.stride + 1)
        #print(f"output_shape = {output_shape}\n")
        for i in range(0, input.shape[-2], self.stride):
            if(i+k > input.shape[-2]):
                break
            for j in range(0, input.shape[-1], self.stride):
                if(j+k > input.shape[-1]):
                    break
                x = torch.flatten(input[:,:, i:i+k, j:j+k], start_dim=1)
                exp_vals = self.qlayer(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output
    
#%% Full-fledged quanvolution with amplitude encoding
class QConv2D_AE(nn.Module):
    def __init__(self,  in_channels: int, kernel_size: int, n_layers: int, stride: int, device=None, dtype=None)->None:
        super(QConv2D_AE, self).__init__()
        self.kernel_size = kernel_size
        self.n_qubits = int(math.ceil(math.log(in_channels * self.kernel_size**2, 2)))
        self.n_layers = n_layers
        self.stride = stride

        # First define a q-node
        weight_shapes = {"weights": (self.n_layers, self.n_qubits, 3)}
        dev = qml.device("lightning.gpu", wires = self.n_qubits) if torch.cuda.is_available() else qml.device("default.qubit", wires = self.n_qubits)
        qnode = qml.QNode(self.quantum_circuit, dev, interface="torch")
        self.qlayer = qml.qnn.TorchLayer(qnode, weight_shapes)

    def quantum_circuit(self, inputs, weights):
        '''
        # Hadamard Layer # Increases complexity and time of training
        for wire in range(n_qubits):
            qml.Hadamard(wires = wire)
        '''
        # Embedding layer
        qml.AmplitudeEmbedding(features = inputs, wires=range(self.n_qubits), normalize=True, pad_with=0.)
        # Variational layer
        for _ in range(self.n_layers):
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_qubits))
            
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(self.n_qubits)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        # Assert the rules
        assert len(input.shape) == 4                                                # (batch_size, n_channels, x_dim, y_dim)
        assert input.shape[-1] == input.shape[-2]
        assert self.stride > 0
        #assert input.shape[1] == 1                                                  # Supports only one channel only

        k = self.kernel_size                                                             # kernel dimension
        output = None
        iterator = 0
        output_shape = int((input.shape[-1]-k)/self.stride + 1)
        #print(f"output_shape = {output_shape}\n")
        for i in range(0, input.shape[-2], self.stride):
            if(i+k > input.shape[-2]):
                break
            for j in range(0, input.shape[-1], self.stride):
                if(j+k > input.shape[-1]):
                    break
                x = torch.flatten(input[:,:, i:i+k, j:j+k], start_dim=1)
                exp_vals = self.qlayer(x).view(input.shape[0], self.n_qubits, 1)
                if(iterator>0):
                    output = torch.cat((output, exp_vals), -1)
                else:
                    output = exp_vals
                iterator +=1
        output = torch.reshape(output, (output.shape[0], output.shape[1], output_shape, output_shape))
        return output

