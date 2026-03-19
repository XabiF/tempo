# Circuit simulation benchmark

This serves as a benchmark to compare different existing simulation methods to compute the expectation value of a given observable and quantum circuit. The chosen circuit is shown [here](circuit.ipynb), where we vary the number of qubits. Average elapsed time and average peak memory usage are compared.

## Technical notes

For fully-TN based methods, contracting a expected-value $\bra{\psi} O \ket{\psi}$ tensor network like this one (which is of constant width $r$ and, of course, of height $n$, where $n$ is the number of qubits), if following one of the most optimal contraction schemes, scales like $\mathcal{O}{(n \exp{(r)})}$, which is clearly reflected in the benchmark plot.

Note that $r$ represents the width of the TN (which is twice the width of the circuit, plus one for the observable MPO) given that the circuit MPOs have constant bond dimensions, or otherwise, serves as an aggregate measure of the TN width and the bond dimension sizes of the MPOs. This is the case since two MPOs of bond dimensions $p$ and $q$ can be "joined" (either numerically contracted or just constructed as a single one in the first place) into a single MPO of bond dimension $pq$, so there is an exponential relationship between the total MPO count and the overall product of bond dimensions.

Thus, this circuit is a good example of the family of circuits efficiently simulable using quantum-inspired TN methods.

Meanwhile, naive statevector-like circuit simulations are known to inefficiently scale as $\mathcal{O}{(\exp{(n)})}$.

## Simulated backends

- Several widely used Qiskit simulator backends

- Several widely used PennyLane device backends

- Manual TN contractions (same optimal contraction scheme)

  - Using **tensorflow**
  - Using **pytorch**
  - Using **numpy**
  - Using these libraries (pure C++)
  - Using these libraries (Python layer)

## Other options (feel free to suggest fixes or alternatives):

- numpy+JAX was left out, since JAX optimizations would constrain too much the MPS-MPO-MPS contraction subroutine
