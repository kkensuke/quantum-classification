import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

INPUT_SCALE = np.pi / 2
SOFTMAX_SCALE = 10


class QuantumClassifier:
    def __init__(
            self,
            inputs,
            outputs,
            nqubits,
            embedding_nlayers,
            ansatz_nlayers,
            embedding_type,
            ansatz_type,
            initialization_type,
            cost_type,
            shots=None,
            stepsize=0.01,
            steps=100,
    ):
        """Initialize a classifier.
        Args:
            inputs (array[float]): array of input data
            outputs (array[int]): array of output data
            nqubits (int): the number of qubits in the circuit
            embedding_nlayers (int): the number of layers of embedding (except for APE)
            ansatz_nlayers (int): the number of layers of ansatz
            embedding_type (str): the type of embedding circuit;
                Tensor Product Embedding (TPE), Hardware Efficient Embedding (HEE),
                Classically Hard Embedding (CHE), Amplitude Embedding (APE).
            ansatz_type (str): the types of ansatz circuit;
                Tensor Product Ansatz (TPA), Hardware Efficient Ansatz (HEA),
                Strongly Entangling Ansatz (SEA).
            initialization_type (str): the types of initialization;
            cost_type (str): the types of cost function; Mean Squared Error (MSE), Cross Entropy (LOG)
            shots (int): the number of shots
            stepsize (float): the stepsize of optimization
            steps (int): the number of steps of optimization
        """
        # there are degrees of freedom in how to convert the inputs into angles
        # np.arcsin(input[i]) or np.arccos(input[i]**2) in ref. QCL
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs).astype(int).ravel()
        self.input_size = len(self.inputs[0])
        self.nlabels = len(set(self.outputs))
        self.nqubits = nqubits
        self.embedding_nlayers = embedding_nlayers
        self.ansatz_nlayers = ansatz_nlayers
        self.embedding_type = embedding_type
        self.ansatz_type = ansatz_type
        self.initialization_type = initialization_type
        self.cost_type = cost_type
        self.shots = shots
        self.stepsize = stepsize
        self.steps = steps
        self.params = None
        
        valid_embedding_types = {"TPE", "ALE", "HEE", "CHE", "MPS", "APE", "NON"}
        valid_ansatz_types = {"TPA", "ALA", "HEA", "SEA"}
        valid_initialization_types = {"zero", "small", "random", "gaussian"}
        valid_cost_types = {"MAE", "MSE", "LOG"}

        if self.embedding_type not in valid_embedding_types:
            raise ValueError(f"Invalid embedding type. Must be one of {valid_embedding_types}")
        if self.ansatz_type not in valid_ansatz_types:
            raise ValueError(f"Invalid ansatz type. Must be one of {valid_ansatz_types}")
        if self.initialization_type not in valid_initialization_types:
            raise ValueError(f"Invalid initialization type. Must be one of {valid_initialization_types}")
        if self.cost_type not in valid_cost_types:
            raise ValueError(f"Invalid cost type. Must be one of {valid_cost_types}")

        if self.embedding_type == "APE" and self.input_size > 2**self.nqubits:
            raise ValueError("inputs_size must be less than or equal to 2^nqubits when embedding_type is APE")


    def TPE(self, input):
        """Tensor Product Embedding"""
        for _ in range(self.embedding_nlayers):
            for i in range(self.nqubits):
                qml.RX(input[i % self.input_size], wires=i)
                qml.RY(input[i % self.input_size], wires=i)

    def ALE(self, input):
        """Alternating Layered Embedding"""
        self.count = 0
        for _ in range(self.embedding_nlayers):
            if self.count % 2 == 0:
                for i in range(self.nqubits // 2):
                    qml.RX(input[(2*i) % self.input_size], 2*i)
                    qml.RY(input[(2*i) % self.input_size], 2*i)
                    qml.RX(input[(2*i + 1) % self.input_size], 2*i + 1)
                    qml.RY(input[(2*i + 1) % self.input_size], 2*i + 1)
                    qml.CZ(wires=[2*i, 2*i + 1])
            else:
                if self.nqubits % 2 == 0:
                    for i in range(self.nqubits // 2 - 1):
                        qml.RX(input[(2*i + 1) % self.input_size], 2*i + 1)
                        qml.RY(input[(2*i + 1) % self.input_size], 2*i + 1)
                        qml.RX(input[(2*(i + 1)) % self.input_size], 2*(i + 1))
                        qml.RY(input[(2*(i + 1)) % self.input_size], 2*(i + 1))
                        qml.CZ(wires=[2*i + 1, 2 * (i + 1)])
                else:
                    for i in range(self.nqubits // 2):
                        qml.RX(input[(2*i + 1) % self.input_size], 2*i + 1)
                        qml.RY(input[(2*i + 1) % self.input_size], 2*i + 1)
                        qml.RX(input[(2*(i + 1)) % self.input_size], 2*(i + 1))
                        qml.RY(input[(2*(i + 1)) % self.input_size], 2*(i + 1))
                        qml.CZ(wires=[2*i + 1, 2 * (i + 1)])
            self.count += 1
    
    def HEE(self, input):
        """Hardware Efficient Embedding"""
        for _ in range(self.embedding_nlayers):
            for i in range(self.nqubits):
                qml.RX(input[i % self.input_size], wires=i)
                qml.RY(input[i % self.input_size], wires=i)
            for i in range(self.nqubits - 1):
                qml.CNOT(wires=[i, i + 1])

    def CHE(self, input):
        """Classically Hard Embedding"""
        for _ in range(self.embedding_nlayers):
            for i in range(self.nqubits):
                qml.Hadamard(wires=i)
                qml.RZ(input[i % self.input_size], wires=i)
            for i in range(self.nqubits - 1):
                for j in range(i + 1, self.nqubits):
                    qml.CNOT(wires=[i, j])
                    qml.RZ(input[i % self.input_size] * input[j % self.input_size], wires=j)
                    qml.CNOT(wires=[i, j])

    def MPS_block(self, weights, wires):
        qml.RX(weights[0], wires=wires[0])
        qml.RY(weights[0], wires=wires[0])
        qml.RX(weights[1], wires=wires[1])
        qml.RY(weights[1], wires=wires[1])
        qml.CNOT(wires=[wires[0], wires[1]])

    def MPS(self, input):
        """Matrix Product State Embedding"""
        n_wires = self.nqubits
        n_block_wires = 2
        n_params_block = 2

        template_weights = []
        for i in range(self.nqubits - 1):
            template_weights.append([input[i % self.input_size], input[(i + 1) % self.input_size]])

        for _ in range(self.embedding_nlayers):
            qml.MPS(range(n_wires), n_block_wires, self.MPS_block, n_params_block, template_weights)

    def APE(self, input):
        """Amplitude Embedding"""
        qml.AmplitudeEmbedding(
            features=input,
            wires=range(self.nqubits),
            pad_with=1,
            normalize=True,
        )

    def embedding(self, input):
        """Embedding templates for the variational circuit.
        Args:
            input(array[float]): input data
        """
        if self.embedding_type == "TPE":
            self.TPE(input)
        elif self.embedding_type == "ALE":
            self.ALE(input)
        elif self.embedding_type == "HEE":
            self.HEE(input)
        elif self.embedding_type == "CHE":
            self.CHE(input)
        elif self.embedding_type == "MPS":
            self.MPS(input)
        elif self.embedding_type == "APE":
            self.APE(input)
        elif self.embedding_type == "NON":
            pass
        else:
            pass
    
    def TPA(self, params):
        """Tensor Product Ansatz"""
        for i in range(self.ansatz_nlayers):
            for j in range(self.nqubits):
                qml.RX(params[self.nqubits * i + j], wires=j)
                qml.RY(params[self.nqubits * i + j], wires=j)
    
    def ALA(self, params):
        """Alternating Layered Ansatz"""
        self.count = 0
        for i in range(self.ansatz_nlayers):
            if self.count % 2 == 0:
                for j in range(self.nqubits // 2):
                    qml.RX(params[self.nqubits*i + 2*j], 2*j)
                    qml.RY(params[self.nqubits*i + 2*j], 2*j)
                    qml.RX(params[self.nqubits*i + 2*j + 1], 2*j + 1)
                    qml.RY(params[self.nqubits*i + 2*j + 1], 2*j + 1)
                    qml.CZ(wires=[2*j, 2*j + 1])
            else:
                if self.nqubits % 2 == 0:
                    for j in range(self.nqubits // 2 - 1):
                        qml.RX(params[self.nqubits*i + 2*j + 1], 2*j + 1)
                        qml.RY(params[self.nqubits*i + 2*j + 1], 2*j + 1)
                        qml.RX(params[self.nqubits*i + 2*(j + 1)], 2*(j + 1))
                        qml.RY(params[self.nqubits*i + 2*(j + 1)], 2*(j + 1))
                        qml.CZ(wires=[2*j + 1, 2 * (j + 1)])
                else:
                    for j in range(self.nqubits // 2):
                        qml.RX(params[self.nqubits*i + 2*j + 1], 2*j + 1)
                        qml.RY(params[self.nqubits*i + 2*j + 1], 2*j + 1)
                        qml.RX(params[self.nqubits*i + 2*(j + 1)], 2*(j + 1))
                        qml.RY(params[self.nqubits*i + 2*(j + 1)], 2*(j + 1))
                        qml.CZ(wires=[2*j + 1, 2 * (j + 1)])
            self.count += 1
    
    def HEA(self, params):
        """Hardware Efficient Ansatz"""
        for i in range(self.ansatz_nlayers):
            for j in range(self.nqubits):
                qml.RX(params[self.nqubits * i + j], wires=j)
                qml.RY(params[self.nqubits * i + j], wires=j)
            for j in range(self.nqubits - 1):
                qml.CNOT(wires=[j, j + 1])
    
    def SEA(self, params):
        """Strongly Entangling Ansatz"""
        qml.StronglyEntanglingLayers(params, wires=range(self.nqubits))
    
    def ansatz(self, params):
        """Ansatz templates for a variational circuit."""
        if self.ansatz_type == "TPA":
            self.TPA(params)
        elif self.ansatz_type == "ALA":
            self.ALA(params)
        elif self.ansatz_type == "HEA":
            self.HEA(params)
        elif self.ansatz_type == "SEA":
            self.SEA(params)
        else:
            pass
    
    def make_initial_params(self):
        """Generate random parameters corresponding to the ansatz_type.
        Returns:
            params (array[float]): array of parameters
        """
        if self.ansatz_type in ("TPA", "ALA", "HEA"):
            param_shape = 2 * self.nqubits * self.ansatz_nlayers
        else:
            param_shape = qml.StronglyEntanglingLayers.shape(self.ansatz_nlayers, n_wires=self.nqubits)

        if self.initialization_type == "zero":
            params = np.zeros(param_shape, requires_grad=True)
        elif self.initialization_type == "small":
            params = np.random.uniform(0, np.pi/self.nqubits/self.ansatz_nlayers, size=param_shape, requires_grad=True)
        elif self.initialization_type == "random":
            params = np.random.uniform(0, 2*np.pi, size=param_shape, requires_grad=True)
        elif self.initialization_type == "gaussian":
            params = np.random.normal(0, np.pi/self.nqubits/self.ansatz_nlayers, size=param_shape, requires_grad=True)
        else:
            raise ValueError("Unknown initialization_type")

        return params

    def make_circuit(self):
        """Generate a variational quantum circuit. Combine embedding and ansatz.
        Returns:
            QuantumCircuit: variational quantum circuit
        """
        dev = qml.device("lightning.qubit", wires=self.nqubits, shots=self.shots)
        # dev = qml.device("default.qubit", wires=self.nqubits, shots=self.shots)

        def func(params, input):
            self.embedding(input)
            qml.Barrier(only_visual=True, wires=range(self.nqubits))
            self.ansatz(params)

            return np.array([qml.expval(qml.PauliZ(wires=self.nqubits - i - 1)) for i in range(self.nlabels)])

        circuit = qml.QNode(func, dev, diff_method="adjoint")
        # circuit = qml.QNode(func, dev)
        return circuit

    @staticmethod
    def softmax(x):
        x = np.array(x)
        x -= x.max(axis=1, keepdims=True)  # to avoid exp overflow
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    @staticmethod
    def relabel(outputs):
        """Relabel the outputs.
        e.g., 1,2,4,5,7 -> 0,1,2,3,4
            -2,-1,0,1,2 -> 0,1,2,3,4
        """
        set_outputs = set(outputs)

        relabel_dict = dict(zip(sorted(list(set_outputs)), range(len(set_outputs))))
        outputs_ = np.array([relabel_dict[x] for x in outputs]).astype(int)
        return outputs_

    def to_one_hot(self, outputs):
        return np.eye(self.nlabels)[self.relabel(outputs)]
    
    def cost(self, params, inputs, outputs):
        """Cost function of the variational circuit.
        Args:
            params (array[float]): array of ansatz parameters
        Returns:
            cost (float)
        """
        circuit = self.make_circuit()
        one_hot_outputs = self.to_one_hot(outputs)
        predictions = self.softmax([SOFTMAX_SCALE * circuit(params, x * INPUT_SCALE) for x in inputs])

        if self.cost_type == "MAE":
            cost = np.mean(np.abs(one_hot_outputs - predictions))
        elif self.cost_type == "MSE":
            cost = np.mean((one_hot_outputs - predictions) ** 2)
        elif self.cost_type == "LOG":
            cost = -np.mean(one_hot_outputs * np.log(predictions) + (1 - one_hot_outputs) * np.log(1 - predictions))
        else:
            raise ValueError("Invalid cost_type")

        return cost

    def optimize(self):
        """Optimize the variational circuit.
        Optimizers:
            Adagrad Optimizer 	Gradient descent optimizer with past-gradient-dependent learning rate in each dimension.
            AdamOptimizer 	Gradient descent optimizer with adaptive learning rate, first and second moment.
            GradientDescentOptimizer 	Basic gradient descent optimizer.
            Momentum Optimizer 	Gradient descent optimizer with momentum
            NesterovMomentumOptimizer 	Gradient descent optimizer with Nesterov momentum.
            QNGOptimizer 	Optimizer with adaptive learning rate, via calculation of the diagonal or block-diagonal approximation to the Fubini-Study metric tensor. (specific to quantum optimization.)
            RMSPropOptimizer 	Root mean squared propagation optimizer.
        """
        opt = qml.AdamOptimizer(stepsize=self.stepsize)
        cost = lambda params: self.cost(params, self.inputs, self.outputs)

        self.params = self.make_initial_params()
        self.cost_list = []
        for _ in range(self.steps):
            self.params, cost_temp = opt.step_and_cost(cost, self.params)
            self.cost_list.append(cost_temp)

        # return self.params, self.cost_list

    def draw_circuit(self, decompose=False):
        params = self.make_initial_params()
        circuit = self.make_circuit()
        if decompose:
            return qml.draw_mpl(circuit, level="device")(params, self.inputs[0])
        else:
            return qml.draw_mpl(circuit)(params, self.inputs[0])

    def plot_cost(self):
        label = f"{self.embedding_type}, {self.ansatz_type}"
        plt.plot(self.cost_list, label=label)
        plt.xlabel("Steps")
        plt.ylabel(f"{self.cost_type}")
        plt.legend()
        plt.show()
    
    def predict(self, x):
        """Predict the label of the input.
        Args:
            x (array[float]): input data
        Returns:
            prediction (int): predicted label
        """
        x = np.array(x)
        circuit = self.make_circuit()
        labels = np.arange(self.nlabels).astype(int)
        prediction = self.softmax([SOFTMAX_SCALE * circuit(self.params, x * INPUT_SCALE)])
        prediction = np.round(prediction).astype(int)
        prediction = prediction @ labels # one-hot to original label
        return prediction

    def accuracy(self, test_inputs, test_outputs):
        """Calculate the accuracy of the predictions.
        Args:
            test_inputs (array[float]): array of test inputs
            test_outputs (array[float]): array of test outputs
        Returns:
            accuracy (float): the accuracy of the predictions
        """
        test_inputs = np.array(test_inputs)
        test_outputs = np.array(test_outputs)
        
        circuit = self.make_circuit()

        labels = np.arange(self.nlabels).astype(int)
        predictions = self.softmax([SOFTMAX_SCALE * circuit(self.params, x * INPUT_SCALE) for x in test_inputs])
        predictions = np.round(predictions).astype(int)
        predictions = predictions @ labels  # one-hot to original label

        test_outputs_relabeled = self.relabel(np.array(test_outputs).astype(int).ravel())

        accuracy = float(np.sum(predictions == test_outputs_relabeled) / len(test_outputs_relabeled))

        return accuracy
