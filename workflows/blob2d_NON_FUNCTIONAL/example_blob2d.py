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
import matplotlib.pyplot as plt


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

    campaign = uq.CampaignDask(name="Blob.")
    print(f"Running in {campaign.campaign_dir}")
    encoder = boutvecma.BOUTEncoder(template_input="models/blob2d/delta_1/BOUT.inp")
    decoder = boutvecma.decoder.Blob2DDecoder(use_peak=False)
    params = {
        "model:Te0": {"type": "float", "min": 0.0, "max": 10, "default": 5.0},
        "model:n0": {"type": "float", "min": 1e17, "max": 1e20, "default": 2e18},
        "model:D_vort": {"type": "float", "min": 1e-8, "max": 1e-5, "default": 1e-6},
        "model:D_n": {"type": "float", "min": 1e-8, "max": 1e-5, "default": 1e-6},
        "model:R_c": {"type": "float", "min": 0.0, "max": 10, "default": 1.5},
    }

    campaign.add_app("1D_conduction", params=params, encoder=encoder, decoder=decoder)

    vary = {
        "model:Te0": chaospy.Uniform(4.0, 6.0),
        "model:n0": chaospy.Uniform(1.5e18, 2.5e18),
    }

    sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=2)
    campaign.set_sampler(sampler)

    campaign.draw_samples()

    run_dirs = campaign.populate_runs_dir()

    print(f"Created run directories: {run_dirs}")

    if args.batch:
        # Example of use on Viking
        cluster = SLURMCluster(
            job_extra=[
                "--job-name=VVUQ",
                "--account=PHYS-YPIRSE-2019",
            ],
            cores=1,
            memory="1 GB",
            processes=1,
            walltime="00:10:00",
            interface="ib0",
        )
        cluster.scale(16)
        print(f"Job script:\n{cluster.job_script()}")
        client = Client(cluster)
    else:
        client = Client(processes=True, threads_per_worker=1)

    print(client)

    cmd = os.path.abspath("build/models/blob2d/blob2d")

    time_start = time.time()
    campaign.apply_for_each_run_dir(
        uq.actions.ExecuteLocal(f"{cmd} -q -q -q -d .", interpret="mpirun -np 4"),
        client,
    )
    client.close()

    time_end = time.time()

    print(f"Finished, took {time_end - time_start}")

    state_filename = os.path.join(campaign.campaign_dir, "campaign_state.json")
    campaign.save_state(state_filename)
    campaign.collate()

    campaign.apply_analysis(
        uq.analysis.PCEAnalysis(
            sampler=sampler,
            qoi_cols=[
                "peak_x",
                "peak_z",
                "peak_v_x",
                "peak_v_z",
                "com_x",
                "com_z",
                "com_v_x",
                "com_v_z",
            ],
        )
    )

    results = campaign.get_last_analysis()

    campaign.save_state(state_filename)

    moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments.png")
    sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first.png")
    distribution_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "distribution.png"
    )

    plt.figure()
    results.plot_moments("vx", xlabel=r"$\rho$", filename=moment_plot_filename)
    results.plot_moments("vz", xlabel=r"$\rho$", filename=moment_plot_filename)
    plt.figure()
    results.plot_sobols_first("vx", xlabel=r"$\rho$", filename=sobols_plot_filename)
    results.plot_sobols_first("vz", xlabel=r"$\rho$", filename=sobols_plot_filename)

    d = campaign.get_collation_result()
    fig, ax = plt.subplots()
    ax.hist(d.vx[0], density=True, bins=50)
    t1 = results.raw_data["output_distributions"]["vx"][49]
    ax.plot(np.linspace(t1.lower, t1.upper), t1.pdf(np.linspace(t1.lower, t1.upper)))
    fig.savefig(distribution_plot_filename)

    print(
        f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}\n\t{distribution_plot_filename}"
    )
