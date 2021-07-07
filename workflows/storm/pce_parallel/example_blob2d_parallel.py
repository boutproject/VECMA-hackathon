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

#    execute = uq.actions.ExecuteLocal(
#        os.path.abspath(
#            STORMPATH+"/storm2d -d . -q -q -q -q |& tee run.log"
#        ),
#    )
    execute = uq.actions.execute_slurm.ExecuteSLURM(
        "slurm_template.sh", "TARGET_DIR"
    )


    actions = uq.actions.Actions(
        uq.actions.CreateRunDirectory(root=campaign_work_dir, flatten=True),
        uq.actions.Encode(encoder),
        execute,
        uq.actions.Decode(decoder)
    )

    campaign.add_app(
        name="Storm-uq.",
        params=params,
        actions=actions
    )

    # Create the sampler
    vary = {
        "mesh:Ly": chaospy.Uniform(4400, 6600),
#        "storm:R_c": chaospy.Uniform(1.3, 1.7),
#        "storm:mu_n0": chaospy.Uniform(0.0005, 0.05),
#        "storm:mu_vort0": chaospy.Uniform(0.0005, 0.05),
    }
    sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1)

    # Associate the sampler with the campaign
    campaign.set_sampler(sampler)

    time_start = time.time()
    campaign.execute().collate(progress_bar=True)
    #campaign.execute(pool=client).collate(progress_bar=True)
#    campaign.apply_for_each_run_dir(
#        uq.actions.execute_slurm.ExecuteSLURM("slurm_template.sh", "TARGET_DIR"),
#        batch_size=16
#    ).start()

    time_end = time.time()

    print(f"Finished, took {time_end - time_start}")

    state_filename = os.path.join(campaign.campaign_dir, "campaign_state.json")
    campaign.save_state(state_filename)
    campaign.collate()

    exit(0)

    campaign.apply_analysis(
        uq.analysis.SCAnalysis(
            sampler=sampler,
            qoi_cols=[
                "peak_x",
                "peak_z",
                "peak_v_x",
                "peak_v_z",
                # "com_x",
                # "com_z",
                # "com_v_x",
                # "com_v_z",
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
