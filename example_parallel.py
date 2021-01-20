#!/usr/bin/env python3

import argparse
import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
from dask.distributed import Client
from dask_jobqueue import SLURMCluster


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EasyVVUQ applied to BOUT++")
    parser.add_argument(
        "--batch",
        "-b",
        help="Run on a batch (SLURM) system",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    campaign = uq.CampaignDask(name="Conduction.")
    encoder = boutvecma.BOUTEncoder(template_input="models/conduction/data/BOUT.inp")
    decoder = boutvecma.BOUTDecoder(variables=["T"])
    params = {
        "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
        "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
        "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
        "T:gauss_centre": {
            "type": "float",
            "min": 0.0,
            "max": 2 * np.pi,
            "default": np.pi,
        },
    }

    campaign.add_app("1D_conduction", params=params, encoder=encoder, decoder=decoder)

    vary = {
        "conduction:chi": chaospy.LogUniform(np.log(1e-2), np.log(1e2)),
        "T:scale": chaospy.Uniform(0.5, 1.5),
        "T:gauss_width": chaospy.Uniform(0.01, 0.4),
        "T:gauss_centre": chaospy.Uniform(0.0, 2 * np.pi),
    }

    sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)
    campaign.set_sampler(sampler)

    campaign.draw_samples()

    run_dirs = campaign.populate_runs_dir()

    print(f"Created run directories: {run_dirs}")

    if args.batch:
        # Not sure we actually want SLURM for this example, only a few
        # seconds per run, on 1 core
        cluster = SLURMCluster(
            job_extra=["#### TODO: SLURM args ####"],
            queue="#### TODO: SLURM queue ####",
            cores=1,
            memory="1 GB",
            processes=1,
        )
        cluster.scale(32)
        client = Client(cluster)
    else:
        client = Client(processes=True, threads_per_worker=1)

    print(client)

    time_start = time.time()
    campaign.apply_for_each_run_dir(
        uq.actions.ExecuteLocal(
            os.path.abspath("build/models/conduction/conduction -q -q -q -d .")
        ),
        client,
    )
    client.close()

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
