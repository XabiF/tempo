import numpy as np
from numpy.typing import NDArray
import copy

import benchcore

#############################

import tensorflow as tf

class MPS:
    def __init__(self, tensors: list[tf.Tensor], transposed: bool = False):
        self.tensors = tensors
        self.transposed = transposed

    def __getitem__(self, i: int) -> tf.Tensor:
        tsr = self.tensors[i]
        if self.transposed:
            if i == 0:
                return tsr
            elif i == len(self.tensors)-1:
                return tsr.transpose(1,0)
            else:
                return tsr.transpose(1,0,2)
        else:
            return tsr
        
    def __len__(self) -> int:
        return len(self.tensors)
        
    def set_transposed(self, transposed: bool):
        self.transposed = transposed

    def as_transposed(self):
        shallow_copy = copy.copy(self)
        shallow_copy.set_transposed(True)
        return shallow_copy

class MPO:
    def __init__(self, tensors: list[tf.Tensor], transposed: bool = False):
        self.tensors = tensors
        self.transposed = transposed

    def __getitem__(self, i: int) -> tf.Tensor:
        tsr = self.tensors[i]
        if self.transposed:
            if i == 0:
                return tsr.transpose(1,0,2)
            elif i == len(self.tensors)-1:
                return tsr.transpose(2,1,0)
            else:
                return tsr.transpose(2,1,0,3)
        else:
            return tsr
        
    def __len__(self) -> int:
        return len(self.tensors)
        
    def set_transposed(self, transposed: bool):
        self.transposed = transposed

    def as_transposed(self):
        shallow_copy = copy.copy(self)
        shallow_copy.set_transposed(True)
        return shallow_copy

def contract_mps_mpos_mps_rowbyrow(start_mps: MPS, mpo_list: list[MPO], end_mps: MPS):
    n = len(start_mps)
    end_mps_transp = end_mps.as_transposed()
    mps_mpo_count = 2 + len(mpo_list)

    # First row
    cur_contr_tsr = start_mps[n-1]
    for i in range(mps_mpo_count-2):
        cur_contr_tsr = tf.tensordot(cur_contr_tsr, mpo_list[i][n-1], axes=([i+1],[0]))
    cur_contr_tsr = tf.tensordot(cur_contr_tsr, end_mps_transp[n-1], axes=([mps_mpo_count-1],[0]))
    
    # Middle rows
    for j in range(1, n-1):
        cur_contr_tsr = tf.tensordot(start_mps[n-1-j], cur_contr_tsr, axes=([2],[0]))
        for i in range(1, mps_mpo_count-1):
            cur_contr_tsr = tf.transpose(tf.tensordot(mpo_list[i-1][n-1-j], cur_contr_tsr, axes=([0,3],[i,i+1])), perm=(list(range(2,2+i)) + [0,1] + list(range(2+i,mps_mpo_count+1))))
        cur_contr_tsr = tf.tensordot(cur_contr_tsr, end_mps_transp[n-1-j], axes=([mps_mpo_count-1,mps_mpo_count],[0,2]))

    # Last row
    cur_contr_tsr = tf.tensordot(start_mps[0], cur_contr_tsr, axes=([1],[0]))
    for i in range(1, mps_mpo_count-1):
        cur_contr_tsr = tf.tensordot(mpo_list[i-1][0], cur_contr_tsr, axes=([0,2],[0,1]))
    cur_contr_tsr = tf.tensordot(cur_contr_tsr, end_mps_transp[0], axes=([0,1],[0,1]))

    return cur_contr_tsr.numpy()

#############################

NUM_THETA_LAYERS = 3
N_RANGE = range(2, 60+1)

def create_zero_state_mps(n: int):
    top_tensor = np.zeros([2, 1])
    top_tensor[0, 0] = 1.0

    middle_tensor = np.zeros([1, 2, 1])
    middle_tensor[0, 0, 0] = 1.0
    middle_tensors = [middle_tensor]*(n-2)

    bottom_tensor = np.zeros([1, 2])
    bottom_tensor[0, 0] = 1.0

    return MPS([top_tensor] + middle_tensors + [bottom_tensor])

def create_theta_ry_mpo(n: int, theta: NDArray, base_index: int):
    c0 = np.cos(theta[base_index+0]/2)
    s0 = np.sin(theta[base_index+0]/2)
    top_tensor = np.zeros([2, 2, 1])
    top_tensor[0, 0, 0] = c0
    top_tensor[0, 1, 0] = s0
    top_tensor[1, 0, 0] = -s0
    top_tensor[1, 1, 0] = c0

    middle_tensors = []
    for j in range(1, n-1):
        cj = np.cos(theta[base_index+j]/2)
        sj = np.sin(theta[base_index+j]/2)
        middle_tensor = np.zeros([2, 1, 2, 1])
        middle_tensor[0, 0, 0, 0] = cj
        middle_tensor[0, 0, 1, 0] = sj
        middle_tensor[1, 0, 0, 0] = -sj
        middle_tensor[1, 0, 1, 0] = cj
        middle_tensors.append(middle_tensor)

    cf = np.cos(theta[base_index+n-1]/2)
    sf = np.sin(theta[base_index+n-1]/2)
    bottom_tensor = np.zeros([2, 1, 2])
    bottom_tensor[0, 0, 0] = cf
    bottom_tensor[0, 0, 1] = sf
    bottom_tensor[1, 0, 0] = -sf
    bottom_tensor[1, 0, 1] = cf

    return MPO([top_tensor] + middle_tensors + [bottom_tensor])

def create_linear_entangl_mpo(n: int):
    top_tensor = np.zeros([2, 2, 2])
    top_tensor[0, 0, 0] = 1
    top_tensor[1, 1, 1] = 1

    middle_tensor = np.zeros([2, 2, 2, 2])
    middle_tensor[0, 0, 0, 0] = 1
    middle_tensor[1, 0, 1, 1] = 1
    middle_tensor[0, 1, 1, 1] = 1
    middle_tensor[1, 1, 0, 0] = 1
    middle_tensors = [middle_tensor]*(n-2)

    bottom_tensor = np.zeros([2, 2, 2])
    bottom_tensor[0, 0, 0] = 1
    bottom_tensor[1, 0, 1] = 1
    bottom_tensor[0, 1, 1] = 1
    bottom_tensor[1, 1, 0] = 1

    return MPO([top_tensor] + middle_tensors + [bottom_tensor])

def create_Z_expval_mpo(n: int):
    top_tensor = np.zeros([2, 2, 1])
    top_tensor[0, 0, 0] = 1
    top_tensor[1, 1, 0] = -1

    middle_tensor = np.zeros([2, 1, 2, 1])
    middle_tensor[0, 0, 0, 0] = 1
    middle_tensor[1, 0, 1, 0] = -1
    middle_tensors = [middle_tensor]*(n-2)

    bottom_tensor = np.zeros([2, 1, 2])
    bottom_tensor[0, 0, 0] = 1
    bottom_tensor[1, 0, 1] = -1

    return MPO([top_tensor] + middle_tensors + [bottom_tensor])

def compute_expval(n: int, theta: NDArray):
    mps = create_zero_state_mps(n)
    mpos = [create_theta_ry_mpo(n, theta, 0)]
    for j in range(1, NUM_THETA_LAYERS):
        mpos.append(create_linear_entangl_mpo(n))
        mpos.append(create_theta_ry_mpo(n, theta, j*n))
    mpos.append(create_Z_expval_mpo(n))
    for j in range(NUM_THETA_LAYERS-1, 0, -1):
        mpos.append(mpos[2*j].as_transposed())
        mpos.append(mpos[2*j-1].as_transposed())
    mpos.append(mpos[0].as_transposed())

    return contract_mps_mpos_mps_rowbyrow(mps, mpos, mps)

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
        expval_results.append((n, "manual-tn-tensorflow", expval, expval_time))
    return theta_results, expval_results
