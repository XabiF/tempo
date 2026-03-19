import numpy as np
from numpy.typing import NDArray

import benchcore

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import EstimatorV2 as Estimator

NUM_THETA_LAYERS = 3
N_RANGE = range(2, 25+1)

def circuit(n: int, theta: NDArray):
    qc = QuantumCircuit(n)

    for i in range(n):
        qc.ry(theta[i], i)

    for j in range(1, NUM_THETA_LAYERS):
        for i in range(n-1):
            qc.cx(i, i+1)
        for i in range(n):
            qc.ry(theta[i + j*n], i)

    return qc

def compute_expval(n: int, theta: NDArray, backend):
    obs = SparsePauliOp("Z"*n)
    qc = circuit(n, theta)

    estimator = Estimator()
    pass_manager = generate_preset_pass_manager(optimization_level=0, backend=backend)
    isa_qc = pass_manager.run(qc)
    pub = (isa_qc, obs)
    job = estimator.run([pub])
    result = job.result()
    return float(result[0].data.evs)

def gen_theta(n):
    theta = np.zeros((n*NUM_THETA_LAYERS))
    for i in range(n*NUM_THETA_LAYERS):
        theta[i] = 2*np.pi*(i / (n*NUM_THETA_LAYERS))
    return theta

def benchmark():
    backends = AerSimulator().available_methods()

    theta_results = []
    expval_results = []
    for n in N_RANGE:
        theta, theta_time = benchcore.study_fn(lambda: gen_theta(n))
        theta_results.append((n, theta_time))

        for backend_name in backends:
            backend = AerSimulator(method=backend_name)
            backend.set_options(noise_model=None)
            expval, expval_time = benchcore.study_fn(lambda: compute_expval(n, theta, backend))
            expval_results.append((n, "qiskit-" + backend_name, expval, expval_time))

    return theta_results, expval_results
