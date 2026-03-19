import numpy as np
from numpy.typing import NDArray

import benchcore

import pennylane as qml

NUM_THETA_LAYERS = 3
N_RANGE_BASE = range(2, 20+1)
N_RANGE_TENSOR = range(2, 60+1)

def compute_expval(n: int, theta: NDArray, backend: str):
    @qml.set_shots(None if backend == "default.tensor" else 1024)
    @qml.qnode(qml.device(backend, wires=n))
    def ansatz():
        for i in range(n):
            qml.RY(theta[i], wires=i)
        
        for j in range(1, NUM_THETA_LAYERS):
            for i in range(n-1):
                qml.CNOT(wires=[i, i+1])
            for i in range(n):
                qml.RY(theta[i + j*n], wires=i)

        op = qml.PauliZ(0)
        for i in range(1, n):
            op = op @ qml.PauliZ(i)
        return qml.expval(op)

    return ansatz

def gen_theta(n):
    theta = np.zeros((n*NUM_THETA_LAYERS))
    for i in range(n*NUM_THETA_LAYERS):
        theta[i] = 2*np.pi*(i / (n*NUM_THETA_LAYERS))
    return theta

def benchmark():
    backends_base = ["default.qubit", "lightning.qubit"]
    backends_tensor = ["default.tensor"]

    theta_results = []
    expval_results = []
    for n in N_RANGE_BASE:
        theta, theta_time = benchcore.study_fn(lambda: gen_theta(n))
        theta_results.append((n, theta_time))

        for backend_name in backends_base:
            expval, expval_time = benchcore.study_fn(lambda: compute_expval(n, theta, backend_name)())
            expval_results.append((n, "pennylane-" + backend_name, expval, expval_time))

    for n in N_RANGE_TENSOR:
        theta, theta_time = benchcore.study_fn(lambda: gen_theta(n))
        theta_results.append((n, theta_time))

        for backend_name in backends_tensor:
            expval, expval_time = benchcore.study_fn(lambda: compute_expval(n, theta, backend_name)())
            expval_results.append((n, "pennylane-" + backend_name, expval, expval_time))

    return theta_results, expval_results
