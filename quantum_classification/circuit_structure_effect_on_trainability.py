import matplotlib.pyplot as plt

from quantum_classifier import *


def train_each_circuit(
    x_train,
    y_train,
    x_test,
    y_test,
    nqubits,
    embedding_nlayers,
    ansatz_nlayers,
    embedding_list,
    ansatz_list,
    cost_type,
    draw=False,
    shots=None,
    stepsize=0.3,
    steps=50,
):
    optimized_cost_acc = []
    cost_all = []

    for embedding_type in embedding_list:
        cost_embedding = []

        for ansatz_type in ansatz_list:
            label = f"{embedding_type}, {ansatz_type}"; print(label)

            opt_circuit = QuantumClassifier(
                x_train,
                y_train,
                nqubits,
                embedding_nlayers,
                ansatz_nlayers,
                embedding_type,
                ansatz_type,
                cost_type,
                shots,
                stepsize,
                steps,
            )

            if draw:
                opt_circuit.draw_circuit()
            else:
                pass

            opt_circuit.optimize()
            cost_embedding.append((label, opt_circuit.cost_list))

            acc = opt_circuit.accuracy(x_test, y_test)
            cost_ = float(opt_circuit.cost_list[-1])
            optimized_cost_acc.append(
                (
                    f"embedding_type: {embedding_type}, ansatz_type: {ansatz_type}",
                    cost_,
                    acc,
                )
            )

        cost_all.append(cost_embedding)
    return optimized_cost_acc, cost_all


def sort_cost_acc(cost_type, optimized_cost_acc):
    optimized_cost = [(x[0], x[1]) for x in optimized_cost_acc]
    optimized_acc = [(x[0], x[2]) for x in optimized_cost_acc]
    optimized_cost = dict(
        sorted(dict(optimized_cost).items(), key=lambda item: item[1])
    )
    optimized_acc = dict(
        sorted(dict(optimized_acc).items(), key=lambda item: item[1], reverse=True)
    )

    print(cost_type)
    for key, value in optimized_cost.items():
        print(f"{key}: cost {value}")

    print("---------------------------------------------------------------")
    for key, value in optimized_acc.items():
        print(f"{key}: accuracy {value}")


def plot_cost(cost_all):
    plt.figure(figsize=(10, 6))
    for i, cost_embedding in enumerate(cost_all):
        ax = plt.subplot(2, 2, i + 1)
        for (label, cost_list) in cost_embedding:
            ax.semilogy(cost_list, label=label)
        ax.set_xlabel("Steps")
        ax.set_ylabel("Cost")
        # ax.set_ylim(0, 1.)
        ax.legend(bbox_to_anchor=(0.99, 0.98), loc="upper right", borderaxespad=0.0)
    plt.show()
