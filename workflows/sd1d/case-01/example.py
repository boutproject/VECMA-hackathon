#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Path to the executable
EXE_PATH="../../../../SD1D/build/sd1d"

encoder = boutvecma.BOUTEncoder(template_input="../../../../SD1D/build/case-01/BOUT.inp")
decoder = boutvecma.SimpleBOUTDecoder(variables=["Ne"])
params = {
    "P:powerflux": {"type": "float", "min": 1e7, "max": 1e8, "default": 2e7},
    "Ne:flux": {"type": "float", "min": 1e23, "max": 1e24, "default": 4e23},
    #"T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    #"T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        #EXE_PATH + " -d . -q -q -q -q |& tee run.log"
        EXE_PATH + " -d . -q -q -q -q"
        #EXE_PATH + " -d . "
    ),
    decoder,
)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)
# campaign.set_app("1D_conduction")

vary = {
        "P:powerflux": chaospy.Uniform(1e7, 1e8),
        "Ne:flux": chaospy.Uniform(1e23, 1e24),
}

sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)
campaign.set_sampler(sampler)

print(f"Code will be evaluated {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

results_df = campaign.get_collation_result()
print(results_df)
results = campaign.analyse(qoi_cols=["Ne"])

moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments.png")
sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first.png")
distribution_plot_filename = os.path.join(
    f"{campaign.campaign_dir}", "distribution.png"
)

fig, ax = plt.subplots()
xvalues = np.arange(len(results.describe("Ne", "mean")))
ax.fill_between(
    xvalues,
    results.describe("Ne", "mean") - results.describe("Ne", "std"),
    results.describe("Ne", "mean") + results.describe("Ne", "std"),
    label="std",
    alpha=0.2,
)
ax.plot(xvalues, results.describe("Ne", "mean"), label="mean")
try:
    ax.plot(xvalues, results.describe("Ne", "1%"), "--", label="1%", color="black")
    ax.plot(xvalues, results.describe("Ne", "99%"), "--", label="99%", color="black")
except RuntimeError:
    pass
ax.grid(True)
ax.set_ylabel(r"$N_e$")
ax.set_xlabel(r"$\rho$")
ax.legend()
fig.savefig(moment_plot_filename)

plt.figure()
results.plot_sobols_first("Ne", xlabel=r"$\rho$", filename=sobols_plot_filename)

print(f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}")
