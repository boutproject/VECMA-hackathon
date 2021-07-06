#!/usr/bin/env python3

import argparse
import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt


CAMPAIGN_NAME = "Conduction."


def refine_sampling_plan(campaign, analysis, number_of_refinements):
    """
    Refine the sampling plan.

    Parameters
    ----------
    number_of_refinements (int)
       The number of refinement iterations that must be performed.

    Returns
    -------
    None. The new accepted indices are stored in analysis.l_norm and the admissible indices
    in sampler.admissible_idx.
    """

    sampler = campaign.get_active_sampler()

    for _ in range(number_of_refinements):
        # compute the admissible indices
        sampler.look_ahead(analysis.l_norm)

        print(f"Code will be evaluated {sampler.n_new_points[-1]} times")
        # run the ensemble
        campaign.execute().collate(progress_bar=True)

        # accept one of the multi indices of the new admissible set
        data_frame = campaign.get_collation_result()
        analysis.adapt_dimension("T", data_frame)
        analysis.save_state(f"{campaign.campaign_dir}/analysis.state")


def plot_grid_2D(campaign, analysis, i, filename="out.pdf"):
    fig = plt.figure(figsize=[12, 4])
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    accepted_grid = campaign.get_active_sampler().generate_grid(analysis.l_norm)
    ax1.plot(accepted_grid[:, 0], accepted_grid[:, 1], "o")
    ax2.plot(accepted_grid[:, 2], accepted_grid[:, 3], "o")
    ax1.set_title(f"iteration {i}")

    fig.tight_layout()
    fig.savefig(filename)


def custom_moments_plot(results, filename, i):
    fig, ax = plt.subplots()
    xvalues = np.arange(len(results.describe("T", "mean")))
    ax.fill_between(
        xvalues,
        results.describe("T", "mean") - results.describe("T", "std"),
        results.describe("T", "mean") + results.describe("T", "std"),
        label="std",
        alpha=0.2,
    )
    ax.plot(xvalues, results.describe("T", "mean"), label="mean")
    try:
        ax.plot(xvalues, results.describe("T", "1%"), "--", label="1%", color="black")
        ax.plot(xvalues, results.describe("T", "99%"), "--", label="99%", color="black")
    except RuntimeError:
        pass
    ax.grid(True)
    ax.set_ylabel("T")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("iteration " + str(i))
    ax.legend()
    fig.savefig(filename)


def first_time_setup():
    encoder = boutvecma.BOUTEncoder(
        template_input="../../models/conduction/data/BOUT.inp"
    )
    # decoder = boutvecma.LogDataBOUTDecoder(variables=["T"])
    decoder = boutvecma.SimpleBOUTDecoder(variables=["T"])
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
    actions = uq.actions.local_execute(
        encoder,
        os.path.abspath(
            "../../build/models/conduction/conduction -q -q -q -q  -d . |& tee run.log"
        ),
        decoder,
        root=".",
    )
    campaign = uq.Campaign(name=CAMPAIGN_NAME, actions=actions, params=params)

    vary = {
        "conduction:chi": chaospy.Uniform(0.2, 4.0),
        "T:scale": chaospy.Uniform(0.5, 1.5),
        "T:gauss_width": chaospy.Uniform(0.5, 1.5),
        "T:gauss_centre": chaospy.Uniform(0.5 * np.pi, 1.5 * np.pi),
    }

    sampler = uq.sampling.SCSampler(
        vary=vary,
        polynomial_order=1,
        quadrature_rule="C",
        sparse=True,
        growth=True,
        midpoint_level1=True,
        dimension_adaptive=True,
    )
    campaign.set_sampler(sampler)

    print(f"Output will be in {campaign.campaign_dir}")

    sampler = campaign.get_active_sampler()

    print(f"Computing {sampler.n_samples} samples")

    time_start = time.time()
    campaign.execute().collate(progress_bar=True)

    # Create an analysis class and run the analysis.
    analysis = create_analysis(campaign)
    campaign.apply_analysis(analysis)
    analysis.save_state(f"{campaign.campaign_dir}/analysis.state")
    plot_grid_2D(campaign, analysis, 0, f"{campaign.campaign_dir}/grid0.png")

    for i in np.arange(1, 10):
        refine_once(campaign, analysis, i)
    time_end = time.time()

    print(f"Finished, took {time_end - time_start}")

    return campaign


def create_analysis(campaign):
    return uq.analysis.SCAnalysis(sampler=campaign.get_active_sampler(), qoi_cols=["T"])


def refine_once(campaign, analysis, iteration):
    refine_sampling_plan(campaign, analysis, 1)
    campaign.apply_analysis(analysis)
    analysis.save_state(f"{campaign.campaign_dir}/analysis.state")

    results = campaign.last_analysis
    plot_grid_2D(
        campaign,
        analysis,
        iteration,
        f"{campaign.campaign_dir}/grid{iteration:02}.png",
    )
    moment_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", f"moments{iteration:02}.png"
    )
    sobols_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", f"sobols_first{iteration:02}.png"
    )
    results.plot_sobols_first(
        "T",
        ylabel=f"iteration{iteration}",
        xlabel=r"$\rho$",
        filename=sobols_plot_filename,
    )
    plt.ylim(0, 1)
    plt.savefig(f"{campaign.campaign_dir}/sobols{iteration:02}.png")

    custom_moments_plot(results, moment_plot_filename, iteration)

    with open(f"{campaign.campaign_dir}/last_iteration", "w") as f:
        f.write(f"{iteration}")


def plot_results(campaign, moment_plot_filename, sobols_plot_filename):
    results = campaign.get_last_analysis()

    results.plot_sobols_first("T", xlabel=r"$\rho$", filename=sobols_plot_filename)

    fig, ax = plt.subplots()
    xvalues = np.arange(len(results.describe("T", "mean")))
    ax.fill_between(
        xvalues,
        results.describe("T", "mean") - results.describe("T", "std"),
        results.describe("T", "mean") + results.describe("T", "std"),
        label="std",
        alpha=0.2,
    )
    ax.plot(xvalues, results.describe("T", "mean"), label="mean")
    try:
        ax.plot(xvalues, results.describe("T", "1%"), "--", label="1%", color="black")
        ax.plot(xvalues, results.describe("T", "99%"), "--", label="99%", color="black")
    except RuntimeError:
        pass
    ax.grid(True)
    ax.set_ylabel("T")
    ax.set_xlabel(r"$\rho$")
    ax.legend()
    fig.savefig(moment_plot_filename)

    print(f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}")


def reload_campaign(directory):
    """Reload a campaign from a directory

    Returns the campaign, analysis, and last iteration number
    """

    campaign = uq.Campaign(
        name=CAMPAIGN_NAME,
        db_location=f"sqlite:///{os.path.abspath(directory)}/campaign.db",
    )
    analysis = create_analysis(campaign)
    analysis.load_state(f"{campaign.campaign_dir}/analysis.state")

    with open(f"{campaign.campaign_dir}/last_iteration", "r") as f:
        iteration = int(f.read())

    return campaign, analysis, iteration


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "conduction_sc",
        description="Adaptive dimension refinement for 1D conduction model",
    )
    parser.add_argument(
        "--restart", type=str, help="Restart previous campaign", default=None
    )
    parser.add_argument(
        "-n", "--refinement-num", type=int, default=1, help="Number of refinements"
    )

    args = parser.parse_args()

    if args.restart is None:
        first_time_setup()
    else:
        campaign, analysis, last_iteration = reload_campaign(args.restart)
        for iteration in range(
            last_iteration + 1, last_iteration + args.refinement_num + 1
        ):
            refine_once(campaign, analysis, iteration)
