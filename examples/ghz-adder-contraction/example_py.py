import time
import tempo.f64 as tn

# Python example, see comments in C++ example (they all equally apply here)

N = 60

start_t = time.perf_counter()

mps = tn.create_ghz_state_mps(N)
mpo_1 = tn.create_adder_mpo(N, 3)
mpo_2 = tn.create_adder_mpo(N, (1 << N) - 1 - 3)
res = tn.contract_mps_mpos_mps_rowbyrow(mps, [mpo_1, mpo_2], mps)

end_t = time.perf_counter()
duration = end_t - start_t

print(f"Result: {res}")
print(f"Time taken: {duration:2e} s")
