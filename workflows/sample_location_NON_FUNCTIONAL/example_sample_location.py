#!/usr/bin/env python3

"""
This example looks at time traces of temperature at two locations

"""


import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

campaign = uq.Campaign(name="Conduction.")
encoder = boutvecma.BOUTEncoder(template_input="../../models/conduction/data/BOUT.inp")

sample_locations = [
    {"variable": "T", "output_name": "T_centre", "x": 0, "y": 50, "z": 0},
    {"variable": "T", "output_name": "T_edge", "x": 0, "y": 10, "z": 0},
]

decoder = boutvecma.SampleLocationBOUTDecoder(sample_locations=sample_locations)
params = {
    "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}

actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        "../../build/models/conduction/conduction -q -q -q -q  -d . |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Time.", actions=actions, params=params)

vary = {
    "T:scale": chaospy.Uniform(0.5, 1.5),
    "T:gauss_centre": chaospy.Uniform(0.0, np.pi),
}

sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)
campaign.set_sampler(sampler)

time_start = time.time()
campaign.execute().collate()
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

campaign.apply_analysis(
    uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=["T_centre", "T_edge"])
)

results = campaign.get_last_analysis()

state_filename = os.path.join(campaign.campaign_dir, "campaign_state.json")
campaign.save_state(state_filename)

plt.figure()
results.plot_moments(
    "T_centre",
    xlabel=r"$\rho$",
    filename=f"{campaign.campaign_dir}/T_centre_moments.png",
)
plt.figure()
results.plot_sobols_first(
    "T_centre",
    xlabel=r"$\rho$",
    filename=f"{campaign.campaign_dir}/T_centre_sobols_first.png",
)
plt.figure()
results.plot_moments(
    "T_edge", xlabel=r"$\rho$", filename=f"{campaign.campaign_dir}/T_edge_moments.png"
)
plt.figure()
results.plot_sobols_first(
    "T_edge",
    xlabel=r"$\rho$",
    filename=f"{campaign.campaign_dir}/T_edge_sobols_first.png",
)
