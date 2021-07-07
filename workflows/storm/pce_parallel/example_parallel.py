#!/usr/bin/env python3

import argparse
import boutvecma
import easyvvuq as uq
import chaospy
import os
from shutil import rmtree
import numpy as np
import time
import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="EasyVVUQ applied to BOUT++"
    )

    parser.add_argument(
        "--batch",
        "-b",
        help="Run on a batch (SLURM) system",
        action="store_true",
        default=False,
    )
    args = parser.parse_args()

    if not args.batch:
        print("Running locally")
        from dask.distributed import Client
    else:
        print("Running using SLURM")
        from dask.distributed import Client
        from dask_jobqueue import SLURMCluster

    work_dir = os.path.dirname(os.path.abspath(__file__))
    campaign_work_dir = os.path.join(work_dir, "example_parallel")
    # clear the target campaign dir
    if os.path.exists(campaign_work_dir):
        rmtree(campaign_work_dir)
    os.makedirs(campaign_work_dir)

    # Set up a fresh campaign called "coffee_pce"
    db_location = "sqlite:///" + campaign_work_dir + "/campaign.db"
    campaign = uq.Campaign(
        name='Storm-uq.',
        db_location=db_location,
        work_dir=campaign_work_dir
    )

    print(f"Running in {campaign.campaign_dir}")

    # Define parameter space
    params = {
        "mesh:Ly": {"type": "float", "min": 4400.0, "max": 6600.0, "default": 5500.0},
        "storm:R_c": {"type": "float", "min": 1.3, "max": 1.7, "default": 1.5},
        "storm:mu_n0": {"type": "float", "min": 0.0005, "max": 0.05, "default": 0.005},
        "storm:mu_vort0": {"type": "float", "min": 0.0005, "max": 0.05, "default": 0.005},
    }
    # Create an encoder, decoder and collater for PCE test app
    STORMPATH="../../../models/storm2d"
    encoder = boutvecma.BOUTEncoder(
        template_input=STORMPATH+"/data/BOUT.inp"
    )

    decoder = boutvecma.StormProfileBOUTDecoder(
        variables=["n","phi","vort"]
    )

    execute = uq.actions.ExecuteLocal(
        os.path.abspath(
            STORMPATH+"/storm2d -d . -q -q -q -q |& tee run.log"
        ),
    )

    actions = uq.actions.Actions(
        uq.actions.CreateRunDirectory(root=campaign_work_dir, flatten=True),
        uq.actions.Encode(encoder),
        execute,
        uq.actions.Decode(decoder)
    )

    if args.batch:
        # Example of use on Viking
        cluster = SLURMCluster(
            job_extra=[
                "--job-name=BOUT++VVUQ",
                "--account=c01-plasma",
                "--qos=standard",
            ],
            cores=128,
            #memory="1 GB",
            processes=128,
            walltime="00:20:00",
            #interface="ib0",
        )
        cluster.scale(128)
        print(f"Job script:\n{cluster.job_script()}")
        client = Client(cluster)
        print(cluster)
        print(cluster.job_script())
    else:
        client = Client(processes=True, threads_per_worker=1)

    print(client)

    # Add the app (automatically set as current app)
    campaign.add_app(
        name="Conduction",
        params=params,
        actions=actions
    )

    # Create the sampler
    vary = {
        "conduction:chi": chaospy.Uniform(0.2, 4.0),
        "T:scale": chaospy.Uniform(0.5, 1.5),
        "T:gauss_width": chaospy.Uniform(0.01, 0.4),
        "T:gauss_centre": chaospy.Uniform(0.0, 2 * np.pi),
    }
    sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=1)

    # Associate the sampler with the campaign
    campaign.set_sampler(sampler)

    time_start = time.time()
    # Run the cases
    # Note: progress bar seems to be non-functional here, only appearing at
    # 100% once everything is finished...
    campaign.execute(pool=client).collate(progress_bar=True)
    client.close()
    if args.batch:
        client.shutdown()
    time_end = time.time()

    print(f"Finished, took {time_end - time_start}")

    results_df = campaign.get_collation_result()
    results = campaign.analyse(qoi_cols=["T"])

###    state_filename = os.path.join(campaign.campaign_dir, "campaign_state.json")
###    campaign.save_state(state_filename)

    moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments.png")
    sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first.png")
    distribution_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "distribution.png"
    )

    plt.figure()
    results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)

    plt.figure()
    results.plot_sobols_first("T", xlabel=r"$\rho$", filename=sobols_plot_filename)
    print(f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}")
