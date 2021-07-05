#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib

# Do not open figures:
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Path to the storm2 directory that contains the storm2d executable and the
# data directory.
# TODO: move storm2d example to the models directory
STORMPATH="../../../models/storm2d"
varlist = ["n","phi","vort"]
encoder = boutvecma.BOUTEncoder(template_input=STORMPATH+"/data/BOUT.inp")
decoder = boutvecma.StormProfileBOUTDecoder(variables=varlist)

def refine_sampling_plan(number_of_refinements):
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
    for i in range(number_of_refinements):
        # compute the admissible indices
        sampler.look_ahead(analysis.l_norm)

        # run the ensemble
        campaign.execute().collate()

        # accept one of the multi indices of the new admissible set
        data_frame = campaign.get_collation_result()
        analysis.adapt_dimension("n", data_frame)


def plot_grid_2D(i, filename="out.pdf"):
    fig = plt.figure(figsize=[12, 4])
    ax1 = fig.add_subplot(
        121
    )  # , xlim=[0.15, 4.05], ylim=[0.45, 1.55], xlabel='conduction:chi', ylabel='T:scale', title='(x1, x2) plane')
    ax2 = fig.add_subplot(
        122
    )  # , xlim=[0.45, 1.55], ylim=[0.45*np.pi, 1.55*np.pi], xlabel='T:gauss_width', ylabel='T:gauss_centre', title='(x3, x4) plane')
    # ax3 = fig.add_subplot(133, xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel='x19', ylabel='x20', title='(x19, x20) plane')

    accepted_grid = sampler.generate_grid(analysis.l_norm)
    ax1.plot(accepted_grid[:, 0], accepted_grid[:, 1], "o")
    ax2.plot(accepted_grid[:, 2], accepted_grid[:, 3], "o")
    ax1.set_title("iteration " + str(i))
    # ax3.plot(accepted_grid[:,18], accepted_grid[:,19], 'o')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def custom_moments_plot(results, var, filename, i):
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
    ax.set_ylabel(var)
    ax.set_xlabel(r"$x$")
    ax.set_title("iteration " + str(i))
    ax.legend()
    fig.savefig(filename)
    plt.close()


params = {
    "mesh:Ly": {"type": "float", "min": 4400.0, "max": 6600.0, "default": 5500.0},
    "storm:R_c": {"type": "float", "min": 1.3, "max": 1.7, "default": 1.5},
    "storm:mu_n0": {"type": "float", "min": 0.0005, "max": 0.05, "default": 0.005},
    "storm:mu_vort0": {"type": "float", "min": 0.0005, "max": 0.05, "default": 0.005},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        STORMPATH+"/storm2d -d . -q -q -q -q |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Storm2d.", actions=actions, params=params)

vary = {
    "mesh:Ly": chaospy.Uniform(4400, 6600),
    "storm:R_c": chaospy.Uniform(1.3, 1.7),
    "storm:mu_n0": chaospy.Uniform(0.0005, 0.05),
    "storm:mu_vort0": chaospy.Uniform(0.0005, 0.05),
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

print(f"Computing {sampler.n_samples} samples")

time_start = time.time()
campaign.execute().collate()

# results = campaign.analyse(qoi_cols=["T"])

# Create an analysis class and run the analysis.
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=varlist)
campaign.apply_analysis(analysis)
plot_grid_2D(0, "grid0.png")

i = 0
sobols_error = 1e6
error_vs_its = []
samples_vs_its = []
while sobols_error > 1e-3:
    i += 1
    refine_sampling_plan(1)
    campaign.apply_analysis(analysis)
    results = campaign.last_analysis
    samples_vs_its.append(sampler.n_samples)

    plot_grid_2D(i, "grid" + str(i) + ".png")
    for var in varlist:
        moment_plot_filename = os.path.join(
            f"{campaign.campaign_dir}", "moments_" + var + "_" + str(i) + ".png"
        )
        sobols_plot_filename = os.path.join(
            f"{campaign.campaign_dir}", "sobols_first_" + var + "_" + str(i) + ".png"
        )
        sobols_second_plot_filename = os.path.join(
            f"{campaign.campaign_dir}", "sobols_second_" + var + "_" + str(i) + ".png"
        )
        distribution_plot_filename = os.path.join(
            f"{campaign.campaign_dir}", "distribution_" + var + "_" + str(i) + ".png"
        )
        plt.figure()
        results.plot_sobols_first(
            var,
            ylabel="iteration" + str(i),
            xlabel=r"$\rho$",
            filename=sobols_plot_filename,
        )
        plt.ylim(0, 1)
        plt.savefig("sobols_" + var + "_" + str(i) + ".png")
        plt.close()

        plt.figure()
        custom_moments_plot(results, var, moment_plot_filename, i)
        plt.close()

    # Prevent overwrite of old fig
    plt.figure("stat_conv").clear()
    analysis.plot_stat_convergence()
    plt.savefig("stat_convergence.png")
    plt.close()

    sobols = analysis.get_sobol_indices("n")

    if i > 1:
        sobols_error = 0
        count = 0
        for j in sobols:
            count += 1
            sobols_error += np.mean(abs(sobols[j] - sobols_last[j]))
        sobols_error = sobols_error / count
        error_vs_its.append(sobols_error)
        print(str(i) + " " + str(sobols_error))
    sobols_last = sobols

    plt.figure()
    plt.plot(error_vs_its)
    plt.xlabel("Iterations")
    plt.ylabel("Summed error")
    plt.savefig("error_vs_iterations.png")
    plt.close()

    plt.figure()
    plt.semilogy(error_vs_its)
    plt.xlabel("Iterations")
    plt.ylabel("Summed error")
    plt.savefig("error_vs_iterations_log.png")
    plt.close()

    plt.figure()
    plt.plot(samples_vs_its)
    plt.xlabel("Iterations")
    plt.ylabel("Samples")
    plt.savefig("samples_vs_iterations.png")
    plt.close()

    plt.figure()
    plt.semilogy(samples_vs_its)
    plt.xlabel("Iterations")
    plt.ylabel("Samples")
    plt.savefig("samples_vs_iterations_log.png")
    plt.close()

    if i > 20:
        break

    # print(results.sobols_second("T"))
    # plt.figure()
    # results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")


for var in varlist:
    results.plot_sobols_first(var, xlabel=r"$\rho$", filename=sobols_plot_filename)

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
    ax.set_ylabel(var)
    ax.set_xlabel(r"$x$")
    ax.legend()
    fig.savefig(moment_plot_filename)
    plt.close()

plt.figure()
plt.plot(error_vs_its)
plt.xlabel("Iterations")
plt.ylabel("Summed error")
plt.savefig("error_vs_iterations.png")
plt.close()

plt.figure()
plt.semilogy(error_vs_its)
plt.xlabel("Iterations")
plt.ylabel("Summed error")
plt.savefig("error_vs_iterations_log.png")
plt.close()

plt.figure()
plt.plot(samples_vs_its)
plt.xlabel("Iterations")
plt.ylabel("Samples")
plt.savefig("samples_vs_iterations.png")
plt.close()

plt.figure()
plt.semilogy(samples_vs_its)
plt.xlabel("Iterations")
plt.ylabel("Samples")
plt.savefig("samples_vs_iterations_log.png")
plt.close()

###plt.figure()
###results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)

###fig, ax = plt.subplots()
###p = results.get_pdf("T")
###print(p)
# ax.hist(d.T[0], density=True, bins=50)
# t1 = results.raw_data["output_distributions"]["T"][49]
# ax.plot(np.linspace(t1.lower, t1.upper), t1.pdf(np.linspace(t1.lower, t1.upper)))
# fig.savefig(distribution_plot_filename)

# d = campaign.get_collation_result()
# fig, ax = plt.subplots()
# ax.hist(d.T[0], density=True, bins=50)
# t1 = results.raw_data["output_distributions"]["T"][49]
# ax.plot(np.linspace(t1.lower, t1.upper), t1.pdf(np.linspace(t1.lower, t1.upper)))
# fig.savefig(distribution_plot_filename)

print(f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}")
