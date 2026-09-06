import pennylane as qml
from pennylane import numpy as np

from .quantum_classifier import QuantumClassifier

INPUT_SCALE = np.pi / 2


class QuantumClassifier_(QuantumClassifier):
    def make_circuit(self):
        """Generate a variational quantum circuit. Combine embedding and ansatz.
        Returns:
            QuantumCircuit: variational quantum circuit
        """
        dev = qml.device("default.qubit", wires=self.nqubits, shots=self.shots)

        @qml.qnode(dev, interface="numpy")
        def circuit(params, input):

            self.embedding(input)
            qml.Barrier(only_visual=True, wires=range(self.nqubits))
            self.ansatz(params)

            return np.array([qml.probs(wires=i) for i in range(self.nqubits)])

        return circuit

    def cost(self, params):
        """Cost function of the variational circuit.
        Args:
            params (array[float]): array of ansatz parameters
        Returns:
            cost (float)
        """
        circuit = self.make_circuit()
        relabeled_outputs = self.relabel(self.outputs)

        predictions = [1 - np.sum(circuit(params, x)[:, 0]) / self.nqubits for x in self.inputs]

        if self.cost_type == "MAE":
            cost = np.mean(np.array([np.abs(l - pd) for (pd, l) in zip(predictions, relabeled_outputs)]))
        elif self.cost_type == "MSE":
            cost = np.mean(np.array([(l - pd) ** 2 for (pd, l) in zip(predictions, relabeled_outputs)]))
        elif self.cost_type == "LOG":
            cost = np.mean(np.array([-l * self.np_log(pd) - (1-l) * self.np_log(1-pd) for (pd, l) in zip(predictions, relabeled_outputs)]))
        else:
            pass

        return cost
    
    def optimize(self):
        """Optimize the variational circuit."""
        circuit = self.make_circuit()
        relabeled_outputs = self.relabel(self.outputs)

        if self.params is None:
            self.params = self.make_initial_params()
        else:
            pass

        opt = qml.AdamOptimizer(self.stepsize)

        self.cost_list = []
        self.diff_list      = []
        for _ in range(self.steps):
            self.params, cost_temp = opt.step_and_cost(self.cost, self.params)
            self.cost_list.append(cost_temp)
            
            predictions = [1 - np.sum(circuit(self.params, x)[:, 0]) / self.nqubits for x in self.inputs]
            each_step_diff_list = np.array([np.abs(l - pd) for (pd, l) in zip(predictions, relabeled_outputs)])
            self.diff_list.append(each_step_diff_list)

    def accuracy(self, test_inputs, test_outputs):
        """Calculate the accuracy of the predictions.
        Args:
            test_inputs (array[float]): array of test inputs
            test_outputs (array[float]): array of test outputs
        Returns:
            accuracy (float): the accuracy of the predictions
        """
        circuit = self.make_circuit()

        predictions = [1 - np.sum(circuit(self.params, x)[:, 0]) / self.nqubits for x in test_inputs]
        predictions = np.round(predictions).astype(int)

        test_outputs_relabeled = self.relabel(np.array(test_outputs).astype(int).ravel())

        accuracy = float(np.sum(predictions == test_outputs_relabeled) / len(test_outputs_relabeled))

        return accuracy
