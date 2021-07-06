#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Do one run with atol=-15 to find value to use as offset in error decoder
encoder = boutvecma.BOUTExpEncoder(
    template_input="../../../models/conduction/data/BOUT.inp"
)
decoder = boutvecma.SimpleBOUTDecoder(variables=["T"])
params = {
    "solver:atol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        "../../../build/models/conduction/conduction -d . -q -q -q -q |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)

vary = {
    "solver:atol": chaospy.Uniform(-16, -14),
}

pord = 0
sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=pord)
campaign.set_sampler(sampler)

print(f"Initial code evaluation: {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)
time_end = time.time()

print(f"Finished initial step, took {time_end - time_start}")

df = campaign.get_collation_result()

# Offset value is df["T"][50], temperature at centre of domain

# Begin UQ loop:
encoder = boutvecma.BOUTExpEncoder(
    template_input="../../../models/conduction/data/BOUT.inp"
)
decoder = boutvecma.AbsLogErrorBOUTDecoder(
    variables=["T"], error_value=float(df["T"][50])
)
params = {
    "solver:atol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        "../../../build/models/conduction/conduction -d . -q -q -q -q |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)

vary = {
    "solver:atol": chaospy.Uniform(-15, 0),
}

pord = 7
sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=pord)
campaign.set_sampler(sampler)

campaign.draw_samples()
print(f"Code will be evaluated {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

df = campaign.get_collation_result()
collocation_points = df["solver:atol"].to_numpy()

results = campaign.analyse(qoi_cols=["T"])

# This is a surrogate for the error
# Labelled T, but the decode is for the error.
s = results.surrogate()
n = 100
f = np.zeros(shape=(n))
cp = np.zeros(shape=(len(collocation_points)))
avec = np.linspace(-15, 0, n)
for ai, a in enumerate(avec):
    f[ai] = s({"solver:atol": a})["T"]

for ai, a in enumerate(collocation_points):
    cp[ai] = s({"solver:atol": a})["T"]

plt.figure()
plt.plot(avec, f)  # ,'b-')
plt.plot(collocation_points, cp, "ro")
plt.legend()
plt.xlabel("atol")
plt.ylabel("Error at final time, centre point")
plt.savefig("E_vs_atol_order_" + str(pord) + ".png")

plt.figure()
plt.plot(avec, 10 ** f)  # ,'b-')
plt.plot(collocation_points, 10 ** cp, "ro")
plt.legend()
plt.xlabel("atol")
plt.ylabel("Error at final time, centre point")
plt.savefig("E_vs_atol_linear_order_" + str(pord) + ".png")
