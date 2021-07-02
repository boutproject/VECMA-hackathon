#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

encoder = boutvecma.BOUTEncoder(template_input="../../models/conduction/data/BOUT.inp")
decoder = boutvecma.LogDataBOUTDecoder(variables=["T"])
params = {
    "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}
actions = uq.actions.local_execute(encoder, os.path.abspath("../../build/models/conduction/conduction -d . |& tee run.log"), decoder)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)
# campaign.set_app("1D_conduction")

vary = {
    "conduction:chi": chaospy.Uniform(0.2, 4.0),
    "T:scale": chaospy.Uniform(0.5, 1.5),
}

sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)
campaign.set_sampler(sampler)

campaign.draw_samples()
print(f"Code will be evaluated {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate()
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

results = campaign.analyse(qoi_cols=["T"])

moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments.png")
sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first.png")
distribution_plot_filename = os.path.join(
    f"{campaign.campaign_dir}", "distribution.png"
)

campaign.save_state(f"{campaign.campaign_dir}/state.json")

fig, ax = plt.subplots()
xvalues = np.arange(len(results.describe("T", 'mean')))
ax.fill_between(xvalues, results.describe("T", 'mean') -
                results.describe("T", 'std'), results.describe("T", 'mean') +
                results.describe("T", 'std'), label='std', alpha=0.2)
ax.plot(xvalues, results.describe("T", 'mean'), label='mean')
try:
    ax.plot(xvalues, results.describe("T", '1%'), '--', label='1%', color='black')
    ax.plot(xvalues, results.describe("T", '99%'), '--', label='99%', color='black')
except RuntimeError:
    pass
ax.grid(True)
ax.set_ylabel("T")
ax.set_xlabel(r"$\rho$")
ax.legend()
fig.savefig(moment_plot_filename)

plt.figure()
results.plot_sobols_first("T", xlabel=r"$\rho$", filename=sobols_plot_filename)

print(
    f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}"
)
