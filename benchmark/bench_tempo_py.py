import numpy as np
from numpy.typing import NDArray

import benchcore

import tempo.f64 as tn

NUM_THETA_LAYERS = 3
N_RANGE = range(2, 60+1)

def create_zero_state_mps(n: int):
    top_tensor = tn.Tensor2.zeroed([2, 1])
    top_tensor[0, 0] = 1.0

    middle_tensor = tn.Tensor3.zeroed([1, 2, 1])
    middle_tensor[0, 0, 0] = 1.0
    middle_tensors = [middle_tensor]*(n-2)

    bottom_tensor = tn.Tensor2.zeroed([1, 2])
    bottom_tensor[0, 0] = 1.0

    return tn.MPS(top_tensor, middle_tensors, bottom_tensor)

def create_theta_ry_mpo(n: int, theta: NDArray, base_index: int):
    c0 = np.cos(theta[base_index+0]/2)
    s0 = np.sin(theta[base_index+0]/2)
    top_tensor = tn.Tensor3.zeroed([2, 2, 1])
    top_tensor[0, 0, 0] = c0
    top_tensor[0, 1, 0] = s0
    top_tensor[1, 0, 0] = -s0
    top_tensor[1, 1, 0] = c0

    middle_tensors = []
    for j in range(1, n-1):
        cj = np.cos(theta[base_index+j]/2)
        sj = np.sin(theta[base_index+j]/2)
        middle_tensor = tn.Tensor4.zeroed([2, 1, 2, 1])
        middle_tensor[0, 0, 0, 0] = cj
        middle_tensor[0, 0, 1, 0] = sj
        middle_tensor[1, 0, 0, 0] = -sj
        middle_tensor[1, 0, 1, 0] = cj
        middle_tensors.append(middle_tensor)

    cf = np.cos(theta[base_index+n-1]/2)
    sf = np.sin(theta[base_index+n-1]/2)
    bottom_tensor = tn.Tensor3.zeroed([2, 1, 2])
    bottom_tensor[0, 0, 0] = cf
    bottom_tensor[0, 0, 1] = sf
    bottom_tensor[1, 0, 0] = -sf
    bottom_tensor[1, 0, 1] = cf

    return tn.MPO(top_tensor, middle_tensors, bottom_tensor)

def create_linear_entangl_mpo(n: int):
    top_tensor = tn.Tensor3.zeroed([2, 2, 2])
    top_tensor[0, 0, 0] = 1
    top_tensor[1, 1, 1] = 1

    middle_tensor = tn.Tensor4.zeroed([2, 2, 2, 2])
    middle_tensor[0, 0, 0, 0] = 1
    middle_tensor[1, 0, 1, 1] = 1
    middle_tensor[0, 1, 1, 1] = 1
    middle_tensor[1, 1, 0, 0] = 1
    middle_tensors = [middle_tensor]*(n-2)

    bottom_tensor = tn.Tensor3.zeroed([2, 2, 2])
    bottom_tensor[0, 0, 0] = 1
    bottom_tensor[1, 0, 1] = 1
    bottom_tensor[0, 1, 1] = 1
    bottom_tensor[1, 1, 0] = 1

    return tn.MPO(top_tensor, middle_tensors, bottom_tensor)

def create_Z_expval_mpo(n: int):
    top_tensor = tn.Tensor3.zeroed([2, 2, 1])
    top_tensor[0, 0, 0] = 1
    top_tensor[1, 1, 0] = -1

    middle_tensor = tn.Tensor4.zeroed([2, 1, 2, 1])
    middle_tensor[0, 0, 0, 0] = 1
    middle_tensor[1, 0, 1, 0] = -1
    middle_tensors = [middle_tensor]*(n-2)

    bottom_tensor = tn.Tensor3.zeroed([2, 1, 2])
    bottom_tensor[0, 0, 0] = 1
    bottom_tensor[1, 0, 1] = -1

    return tn.MPO(top_tensor, middle_tensors, bottom_tensor)

def compute_expval(n: int, theta: NDArray):
    mps = create_zero_state_mps(n)
    mpos = [create_theta_ry_mpo(n, theta, 0)]
    for j in range(1, NUM_THETA_LAYERS):
        mpos.append(create_linear_entangl_mpo(n))
        mpos.append(create_theta_ry_mpo(n, theta, j*n))
    mpos.append(create_Z_expval_mpo(n))
    for j in range(NUM_THETA_LAYERS-1, 0, -1):
        mpos.append(mpos[2*j].transpose())
        mpos.append(mpos[2*j-1].transpose())
    mpos.append(mpos[0].transpose())

    return tn.contract_mps_mpos_mps_rowbyrow(mps, mpos, mps)

def gen_theta(n):
    theta = np.zeros((n*NUM_THETA_LAYERS))
    for i in range(n*NUM_THETA_LAYERS):
        theta[i] = 2*np.pi*(i / (n*NUM_THETA_LAYERS))
    return theta

def benchmark():
    theta_results = []
    expval_results = []
    for n in N_RANGE:
        theta, theta_time = benchcore.study_fn(lambda: gen_theta(n))
        theta_results.append((n, theta_time))

        expval, expval_time = benchcore.study_fn(lambda: compute_expval(n, theta))
        expval_results.append((n, "manual-tn-tempo-py", expval, expval_time))
    return theta_results, expval_results
