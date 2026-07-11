import numpy as np
import tempo.f64 as tn
import time

# Python example, see comments in C++ example (they all equally apply here)

place_names = ("Bree", "Edoras", "Isengard", "Lórien", "Minas Tirith", "Pelargir", "Rivendel", "Tharbad", "Valle")

priorities_vec = np.array([
    15,
    150,
    35,
    75,
    170,
    50,
    40,
    5,
    15
], dtype=np.float64)

visit_times_vec = np.array([
    3,
    5,
    4,
    4,
    7,
    3,
    5,
    2,
    4
], dtype=np.float64)

N = len(priorities_vec)

SPEED = 9.6

travel_times_mat = np.array([
    [ 0, 200, 150, 140, 285, 315, 100, 67, 225 ],
    [ 200, 0, 48, 100, 102, 117, 172, 133, 235 ],
    [ 150, 48, 0, 83, 150, 163, 135, 83, 225 ],
    [ 140, 100, 83, 0, 158, 192, 77, 100, 145 ],
    [ 285, 102, 150, 158, 0, 43, 200, 229, 245 ],
    [ 315, 117, 163, 192, 43, 0, 243, 252, 290 ],
    [ 100, 172, 135, 77, 200, 243, 0, 100, 125 ],
    [ 67, 133, 83, 100, 229, 252, 100, 0, 220 ],
    [ 225, 235, 225, 145, 245, 290, 125, 220, 0 ]
], dtype=np.float64) / SPEED

root_travel_times_vec = np.array([
    40,
    225,
    175,
    183,
    321,
    342,
    167,
    90,
    270
], dtype=np.float64) / SPEED

xi = 6

t_max = max(np.max(root_travel_times_vec), np.max(travel_times_mat))
c_p = 0.1
c_tt = c_p*(300/t_max)
c_vt = c_tt

tau = 0.9

def compute_solution_cost(sol):
    prio = sum(priorities_vec[sol[i]] for i in range(xi))
    travel_time = root_travel_times_vec[sol[0]] + sum(travel_times_mat[sol[i],sol[i+1]] for i in range(xi-1)) + root_travel_times_vec[sol[xi-1]]
    visit_time = sum(visit_times_vec[sol[i]] for i in range(xi))
    return -c_p*prio + c_tt*travel_time + c_vt*visit_time

def create_problem_cost_mps():
    top_tensor = tn.Tensor2.zeroed([N, N])
    for y0 in range(N):
        top_tensor[y0,y0] = np.exp(-tau*(-c_p*priorities_vec[y0] + c_vt*visit_times_vec[y0] + c_tt*root_travel_times_vec[y0]))

    middle_tensors = []
    for i in range(1, xi-1):
        middle_tensor_i = tn.Tensor3.zeroed([N, N, N])
        for yim1 in range(N):
            for yi in range(N):
                middle_tensor_i[yim1,yi,yi] = np.exp(-tau*(-c_p*priorities_vec[yi] + c_vt*visit_times_vec[yi] + c_tt*travel_times_mat[yim1,yi]))
        middle_tensors.append(middle_tensor_i)

    bottom_tensor = tn.Tensor2.zeroed([N, N])
    for yxm2 in range(N):
        for yxm1 in range(N):
            bottom_tensor[yxm2,yxm1] = np.exp(-tau*(-c_p*priorities_vec[yxm1] + c_vt*visit_times_vec[yxm1] + c_tt*root_travel_times_vec[yxm1] + c_tt*travel_times_mat[yxm2,yxm1]))

    return tn.MPS(top_tensor, middle_tensors, bottom_tensor)

def create_constraint_filter_mpo(k):
    top_tensor = tn.Tensor3.zeroed([N, N, 2])
    for y0 in range(N):
        s = int(y0 == k)
        top_tensor[y0,y0,s] = 1

    middle_tensor = tn.Tensor4.zeroed([N, 2, N, 2])
    for yi in range(N):
        if yi == k:
            middle_tensor[k,0,k,1] = 1
        else:
            middle_tensor[yi,0,yi,0] = 1
            middle_tensor[yi,1,yi,1] = 1
    middle_tensors = [middle_tensor]*(xi-2)

    bottom_tensor = tn.Tensor3.zeroed([N, 2, N])
    for ylast in range(N):
        bottom_tensor[ylast,0,ylast] = 1
        if ylast != k:
            bottom_tensor[ylast,1,ylast] = 1

    return tn.MPO(top_tensor, middle_tensors, bottom_tensor)

print("Solving Hobbit routing problem...")

start_t = time.perf_counter()
##
optim_cost_mps = create_problem_cost_mps()
optim_constraint_filter_mpos = [create_constraint_filter_mpo(k) for k in range(N)]
optim_space_mps = tn.join_mps_mpos(optim_cost_mps, optim_constraint_filter_mpos)
optim_solution = tn.mps_argmax_ptraced(optim_space_mps)
##
end_t = time.perf_counter()

duration = end_t - start_t
print(f"Solved! elapsed time: {duration:2e}s")

print("Optimal route:")
print("-- Hobbiton")
for yi in optim_solution:
    print(f"-- {place_names[yi]}")
print("-- Hobbiton")

print(f"Route cost: {compute_solution_cost(optim_solution)}")
