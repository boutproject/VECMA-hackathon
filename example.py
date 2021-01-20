#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time

campaign = uq.Campaign(name="Conduction.")
encoder = boutvecma.BOUTEncoder(template_input="models/conduction/data/BOUT.inp")
decoder = boutvecma.BOUTDecoder(variables=["T"])
params = {
    "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}

campaign.add_app("1D_conduction", params=params, encoder=encoder, decoder=decoder)

vary = {
    "conduction:chi": chaospy.LogUniform(np.log(1e-2), np.log(1e2)),
    "T:scale": chaospy.Uniform(0.5, 1.5),
}

sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)
campaign.set_sampler(sampler)

campaign.draw_samples()

run_dirs = campaign.populate_runs_dir()

print(f"Created run directories: {run_dirs}")

time_start = time.time()
campaign.apply_for_each_run_dir(
    uq.actions.ExecuteLocal(os.path.abspath("build/models/conduction/conduction -d ."))
)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

campaign.collate()

campaign.apply_analysis(uq.analysis.PCEAnalysis(sampler=sampler, qoi_cols=["T"]))

results = campaign.get_last_analysis()

state_filename = os.path.join(campaign.campaign_dir, "campaign_state.json")
campaign.save_state(state_filename)

results.plot_moments(
    "T", xlabel=r"$\rho$", filename=f"{campaign.campaign_dir}/moments.png"
)
results.plot_sobols_first(
    "T", xlabel=r"$\rho$", filename=f"{campaign.campaign_dir}/sobols_first.png"
)
