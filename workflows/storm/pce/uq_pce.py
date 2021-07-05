#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Path to the storm2 directory that contains the storm2d executable and the
# data directory.
# TODO: move storm2d example to the models directory
STORMPATH="../../../models/storm2d"
encoder = boutvecma.BOUTEncoder(template_input=STORMPATH+"/data/BOUT.inp")
decoder = boutvecma.StormProfileBOUTDecoder(variables=["n","phi","vort"])
params = {
    "mesh:Ly": {"type": "float", "min": 4400.0, "max": 6600.0, "default": 5500.0},
    "storm:R_c": {"type": "float", "min": 1.3, "max": 1.7, "default": 1.5},
    "storm:mu_n0": {"type": "float", "min": 0.0005, "max": 0.05, "default": 0.005},
    "storm:mu_vort0": {"type": "float", "min": 0.0005, "max": 0.05, "default": 0.005},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        STORMPATH+"/storm2d -d . -q -q -q -q |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Storm2d.", actions=actions, params=params)
# campaign.set_app("1D_conduction")

vary = {
    "mesh:Ly": chaospy.Uniform(4400, 6600),
    "storm:R_c": chaospy.Uniform(1.3, 1.7),
    "storm:mu_n0": chaospy.Uniform(0.0005, 0.05),
    "storm:mu_vort0": chaospy.Uniform(0.0005, 0.05),
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
results = campaign.analyse(qoi_cols=["n","phi","vort"])

for var in ["n", "phi", "vort"]:
    moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments_"+var+".png")
    sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first_"+var+".png")
    distribution_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "distribution_"+var+".png"
    )

    # Campaign no longer an attribute save_state
    # campaign.save_state(f"{campaign.campaign_dir}/state.json")

    fig, ax = plt.subplots()
    xvalues = np.arange(len(results.describe(var, "mean")))
    ax.fill_between(
        xvalues,
        results.describe(var, "mean") - results.describe(var, "std"),
        results.describe(var, "mean") + results.describe(var, "std"),
        label="std",
        alpha=0.2,
    )
    ax.plot(xvalues, results.describe(var, "mean"), label="mean")
    try:
        ax.plot(xvalues, results.describe(var, "1%"), "--", label="1%", color="black")
        ax.plot(xvalues, results.describe(var, "99%"), "--", label="99%", color="black")
    except RuntimeError:
        pass
    ax.grid(True)
    ax.set_ylabel(r"$\log("+var+")$")
    ax.set_xlabel(r"$x$")
    ax.legend()
    fig.savefig(moment_plot_filename)

    plt.figure()
    results.plot_sobols_first(var, xlabel=r"$x$", filename=sobols_plot_filename)

    print(f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}")
