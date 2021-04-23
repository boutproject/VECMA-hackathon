#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

encoder = boutvecma.BOUTEncoder(template_input="models/conduction/data/BOUT.inp")
decoder = boutvecma.LogDataBOUTDecoder(variables=["T"])
params = {
    "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}
actions = uq.actions.local_execute(encoder, os.path.abspath("build/models/conduction/conduction -d . |& tee run.log"), decoder)
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

plt.figure()
results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)
plt.figure()
results.plot_sobols_first("T", xlabel=r"$\rho$", filename=sobols_plot_filename)

# d = campaign.get_collation_result()
# fig, ax = plt.subplots()
# ax.hist(d.T[0], density=True, bins=50)
# t1 = results.raw_data["output_distributions"]["T"][49]
# ax.plot(np.linspace(t1.lower, t1.upper), t1.pdf(np.linspace(t1.lower, t1.upper)))
# fig.savefig(distribution_plot_filename)

print(
    f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}"
)
