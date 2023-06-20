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
        def func(params, input):

            self.embedding(input)
            qml.Barrier(only_visual=True, wires=range(self.nqubits))
            self.ansatz(params)

            return [qml.probs(wires=self.nqubits - i - 1) for i in range(self.nlabels)]

        circuit = qml.QNode(func, dev)
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

        predictions = [1 - np.sum( circuit(params, x)[:,0] )/self.nqubits for x in self.inputs]

        if self.cost_type == "MSE":
            cost = np.mean(np.array( [(l - pd) ** 2 for (pd, l) in zip(predictions, relabeled_outputs)] ))
        elif self.cost_type == "LOG":
            cost = np.mean(np.array( [ - l * self.np_log(pd) for (pd, l) in zip(predictions, relabeled_outputs) ] ))
        else:
            pass

        return cost

    def accuracy(self, test_inputs, test_outputs):
        """Calculate the accuracy of the predictions by the circuit.
        Returns:
            accuracy (float): the accuracy of the predictions
        """
        circuit = self.make_circuit()

        labels = np.arange(self.nlabels).astype(int)
        predictions = [ 1 - np.sum( circuit(self.params, x)[:,0] ) / self.nqubits for x in test_inputs ]
        predictions = np.round(predictions).astype(int)

        test_outputs_relabeled = self.relabel(
            np.array(test_outputs).astype(int).ravel()
        )

        accuracy = float(
            np.sum(predictions == test_outputs_relabeled) / len(test_outputs_relabeled)
        )

        return accuracy
