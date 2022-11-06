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
        cost_type,
        shots=None,
        stepsize=0.1,
        steps=100,
    ):
        """Initialize the classifier.
        Args:
            inputs (array[float]): array of input data
            outputs (array[int]): array of output data
            nqubits (int): the number of qubits in the circuit
            embedding_nlayers (int): the number of layers of embedding (except for APE)
            ansatz_nlayers (int): the number of layers of ansatz
            embedding_type (str): type of embedding circuit
            ansatz_type (str): the types of ansatz circuit; Tensor Product Embedding (TPE), Hardware Efficient Embedding (HEE),
                                                            Classically Hard Embedding (CHE), Amplitude Embedding (APE)
            cost_type (str): the types of cost function; Mean Squared Error (MSE), Cross Entropy (LOG)
            shots (int): the number of shots
            stepsize (float): the stepsize of optimization
            steps (int): the number of steps of optimization
        """
        # there are degrees of freedom in how to convert the inputs into angles (np.arcsin(input[i]) or np.arccos(input[i]**2) ref. QCL)
        self.inputs = np.array(inputs * INPUT_SCALE)
        self.outputs = np.array(outputs).astype(int).ravel()
        self.input_size = len(self.inputs[0])
        self.nlabels = len(set(self.outputs))
        self.nqubits = nqubits
        self.embedding_nlayers = embedding_nlayers
        self.ansatz_nlayers = ansatz_nlayers
        self.embedding_type = embedding_type
        self.ansatz_type = ansatz_type
        self.cost_type = cost_type
        self.shots = shots
        self.stepsize = stepsize
        self.steps = steps
        self.params = None

        if (
            self.embedding_type == "TPE"
            or self.embedding_type == "HEE"
            or self.embedding_type == "CHE"
            or self.embedding_type == "APE"
            or self.embedding_type == "NON"
        ):
            pass
        else:
            raise ValueError("Input the correct embedding type")


        if (
            self.ansatz_type == "TPA"
            or self.ansatz_type == "HEA"
            or self.ansatz_type == "SEA"
        ):
            pass
        else:
            raise ValueError("Input the correct ansatz type")


        if (
            self.ansatz_type == "TPA"
            or self.ansatz_type == "HEA"
            or self.ansatz_type == "SEA"
        ):
            if self.input_size <= self.nqubits:
                pass
            else:
                raise ValueError("inputs_size must be less than or equal to  nqubits when ansatz_type is TPA, HEA, or SEA")
        elif self.ansatz_type == "APE":
            if self.input_size <= 2**self.nqubits:
                pass
            else:
                raise ValueError("inputs_size must be less than or equal to 2^nqubits when ansatz_type is APE")
        else:
            pass


        if cost_type == "MSE" or cost_type == "LOG":
            pass
        else:
            raise ValueError("cost_type must be MSE or LOG")

    def embedding(self, input):
        """Embedding templates for the variational circuit.
        Args:
            input(array[float]): input data
        """

        if self.embedding_type == "TPE":
            for _ in range(self.embedding_nlayers):
                for i in range(self.input_size):
                    qml.RX(input[i], wires=i)
                    qml.RY(input[i], wires=i)
        elif self.embedding_type == "HEE":
            for _ in range(self.embedding_nlayers):
                for i in range(self.input_size):
                    qml.RX(input[i], wires=i)
                for i in range(self.input_size - 1):
                    qml.CNOT(wires=[i, i + 1])
        elif self.embedding_type == "CHE":
            for _ in range(self.embedding_nlayers):
                for i in range(self.input_size):
                    qml.Hadamard(wires=i)
                    qml.RZ(input[i], wires=i)
                for i in range(self.input_size - 1):
                    for j in range(i + 1, self.input_size):
                        qml.CNOT(wires=[i, j])
                        qml.RZ(input[i] * input[j], wires=j)
                        qml.CNOT(wires=[i, j])
        elif self.embedding_type == "APE":
            qml.AmplitudeEmbedding(
                features=input,
                wires=range(self.input_size),
                pad_with=1,
                normalize=True,
            )
        elif self.embedding_type == "NON":
            pass
        else:
            pass

    def make_initial_params(self):
        """Generate random parameters corresponding to the ansatz_type.
        Returns:
            params (array[float]): array of parameters
        """
        if self.ansatz_type == "TPA":
            params = np.random.uniform(0, np.pi, size=self.nqubits * self.ansatz_nlayers)
        elif self.ansatz_type == "HEA":
            params = np.random.uniform(0, np.pi, size=self.nqubits * self.ansatz_nlayers)
        elif self.ansatz_type == "SEA":
            shape = qml.StronglyEntanglingLayers.shape(self.ansatz_nlayers, n_wires=self.nqubits)
            params = np.random.random(size=shape)
        else:
            pass
        return params

    def ansatz(self, params):
        """Ansatz templates for the variational circuit."""

        if self.ansatz_type == "TPA":
            for i in range(self.ansatz_nlayers):
                for j in range(self.nqubits):
                    qml.RX(params[self.nqubits * i + j], wires=j)
                    qml.RY(params[self.nqubits * i + j], wires=j)
        elif self.ansatz_type == "HEA":
            for i in range(self.ansatz_nlayers):
                for j in range(self.nqubits):
                    qml.RX(params[self.nqubits * i + j], wires=j)
                    qml.RY(params[self.nqubits * i + j], wires=j)
                for j in range(self.nqubits - 1):
                    qml.CNOT(wires=[j, j + 1])
        elif self.ansatz_type == "SEA":
            qml.StronglyEntanglingLayers(params, wires=range(self.nqubits))
        else:
            pass

    def make_circuit(self):
        """Generate a variational quantum circuit. Combine embedding and ansatz.
        Returns:
            QuantumCircuit: variational quantum circuit
        """
        dev = qml.device("default.qubit", wires=self.nqubits, shots=self.shots)

        def func(params, input):

            self.embedding(input)
            qml.Barrier(only_visual=True, wires=range(self.nqubits))
            self.ansatz(params)

            return [ qml.expval(qml.PauliZ(wires=i)) for i in range(self.nlabels) ]

        circuit = qml.QNode(func, dev)
        return circuit

    def softmax(self, x):  # avoid exp overflow
        x = np.array(x)
        x -= x.max(axis=1, keepdims=True)
        x_exp = np.exp(x)
        return x_exp / np.sum(x_exp, axis=1, keepdims=True)

    def np_log(self, x):  # avoid log(0)
        return np.log(np.clip(a=x, a_min=1e-10, a_max=1e10))

    def relabel(self, outputs):
        """Relabel the outputs.
        i.e., 1,2,4,5,7 -> 0,1,2,3,4
            -2,-1,0,1,2 -> 0,1,2,3,4
        """
        set_outputs = set(self.outputs)

        relabel_dict = dict(
            zip(sorted(list(set_outputs)), range(len(set_outputs)))
        )
        outputs_ = np.array( [ relabel_dict[x] for x in outputs ] ).astype(int)
        return outputs_

    def one_hot(self):
        return np.eye(self.nlabels)[self.relabel(self.outputs)]

    def cost(self, params):
        """Cost function of the variational circuit.
        Args:
            params (array[float]): array of parameters
        Returns:
            cost (float)
        """
        circuit = self.make_circuit()
        one_hot_outputs = self.one_hot()

        # Since using all data, it takes long. Seems better to split into batches
        predictions = self.softmax( [SOFTMAX_SCALE * circuit(params, x) for x in self.inputs] )

        results = []

        if self.cost_type == "MSE":
            for (pd, l) in zip(predictions, one_hot_outputs):
                # mulitple by l[j] to make it similar to the cross entropy cost
                results.append(
                    np.sum( [ l[j] * (l[j] - pd[j]) ** 2 for j in range(self.nlabels) ] )
                )
        elif self.cost_type == "LOG":
            for (pd, l) in zip(predictions, one_hot_outputs):
                results.append(
                    -np.sum( [ l[j] * self.np_log(pd[j]) for j in range(self.nlabels) ] )
                )
        else:
            pass

        cost = np.mean(np.array(results))
        return cost

    def optimize(self):
        """Optimize the variational circuit."""

        if self.params is None:
            self.params = self.make_initial_params()
        else:
            pass

        opt = qml.AdamOptimizer(self.stepsize)

        self.cost_list = []
        for _ in range(self.steps):
            self.params, cost_temp = opt.step_and_cost( self.cost, self.params )
            self.cost_list.append(cost_temp)

        # return self.params, self.cost_list

    def draw_circuit(self):
        params = self.make_initial_params()
        circuit = self.make_circuit()
        fig = qml.draw_mpl(circuit, expansion_strategy="device")( params, self.inputs[0])
        plt.show()

    def plot_cost(self):
        label = f"{self.embedding_type}, {self.ansatz_type}"
        plt.semilogy(self.cost_list, label=label)
        plt.xlabel("Steps")
        plt.ylabel("Cost")
        plt.legend()
        plt.show()

    def accuracy(self, test_inputs, test_outputs):
        """Calculate the accuracy of the predictions by the circuit.
        Returns:
            accuracy (float): the accuracy of the prediction
        """
        circuit = self.make_circuit()

        labels = np.arange(self.nlabels).astype(int)
        predictions = self.softmax( [ SOFTMAX_SCALE * circuit(self.params, x) for x in test_inputs * INPUT_SCALE ] )
        predictions = np.round(predictions).astype(int)
        predictions = predictions @ labels  # one-hot to original label

        test_outputs_relabeled = self.relabel(
            np.array(test_outputs).astype(int).ravel()
        )

        accuracy = float(
            np.sum(predictions == test_outputs_relabeled) / len(test_outputs_relabeled)
        )

        return accuracy
