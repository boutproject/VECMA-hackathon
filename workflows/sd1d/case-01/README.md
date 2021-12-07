# Case 1 Workflows

## example.py

Confidence intervals and Sobol indices using 3rd order PCE.

## sc_adaptive.py

Adaptive stochastic collocation with 4 uncertain parameters.
This is an iterative approach that at each step refines the parameter space
grid in the direction of the parameter that is ``least well understood''.

In this implementation, we use the convergence of Sobol indices as a stopping
criterion. I don't have better ideas, but in some cases
this is a bad criterion, as the convergence of Sobol indices (say in summed
error between iterations) is non-monotonic.
