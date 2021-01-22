#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt


class LogDataBOUTDecoder(boutvecma.BaseBOUTDecoder):
    """Returns log(variable)"""

    def __init__(self, target_filename=None, variables=None):
        """
        Parameters
        ==========
        variables: iterable or None
            Iterable of variables to collect from the output. If None, return everything
        """
        super().__init__(target_filename=target_filename)

        self.variables = variables

    def parse_sim_output(self, run_info=None, *args, **kwargs):
        df = self.get_outputs(run_info)

        return {
            variable: boutvecma.decoder.flatten_dataframe_for_JSON(
                np.log(df[variable][-1, ...])
            )
            for variable in self.variables
        }

    @staticmethod
    def element_version():
        return "0.1.0"


campaign = uq.Campaign(name="Conduction.")
encoder = boutvecma.BOUTEncoder(template_input="models/conduction/data/BOUT.inp")
decoder = LogDataBOUTDecoder(variables=["T"])
params = {
    "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}

campaign.add_app("1D_conduction", params=params, encoder=encoder, decoder=decoder)

vary = {
    "conduction:chi": chaospy.Uniform(0.2, 4.0),
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

moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments.png")
sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first.png")

plt.figure()
results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)
plt.figure()
results.plot_sobols_first("T", xlabel=r"$\rho$", filename=sobols_plot_filename)

print(f"Results are in {moment_plot_filename} and {sobols_plot_filename}")
