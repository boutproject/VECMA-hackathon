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

# Path to the executable
EXE_PATH="../../../../SD1D/build/sd1d"

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
        analysis.adapt_dimension("Ne", data_frame)


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


def custom_moments_plot(results, filename, i):
    fig, ax = plt.subplots()
    xvalues = np.arange(len(results.describe("Ne", "mean")))
    ax.fill_between(
        xvalues,
        results.describe("Ne", "mean") - results.describe("Ne", "std"),
        results.describe("Ne", "mean") + results.describe("Ne", "std"),
        label="std",
        alpha=0.2,
    )
    ax.plot(xvalues, results.describe("Ne", "mean"), label="mean")
    try:
        ax.plot(xvalues, results.describe("Ne", "1%"), "--", label="1%", color="black")
        ax.plot(xvalues, results.describe("Ne", "99%"), "--", label="99%", color="black")
    except RuntimeError:
        pass
    ax.grid(True)
    ax.set_ylabel("Ne")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("iteration " + str(i))
    ax.legend()
    fig.savefig(filename)
    plt.close()


encoder = boutvecma.BOUTEncoder(template_input="../../../../SD1D/build/case-01/BOUT.inp")
decoder = boutvecma.SimpleBOUTDecoder(variables=["Ne"])
params = {
    "P:powerflux": {"type": "float", "min": 1e7, "max": 1e8, "default": 2e7},
    "Ne:flux": {"type": "float", "min": 1e22, "max": 1e24, "default": 4e23},
    "sd1d:gamma_sound": {"type": "float", "min": 0, "max": 10.0/3.0, "default": 5.0/3.0},
    "sd1d:nloss": {"type": "float", "min": 0, "max": 1e5, "default": 1e3},
    #"T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    #"T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        EXE_PATH + " -d . -q -q -q -q"
    ),
    decoder,
)
campaign = uq.Campaign(name="adaptive_sc.", actions=actions, params=params)

vary = {
        "P:powerflux": chaospy.Uniform(1e7, 1e8),
        "Ne:flux": chaospy.Uniform(1e23, 1e24),
        "sd1d:gamma_sound": chaospy.Uniform(1.0, 7.0/3.0),
        "sd1d:nloss": chaospy.Uniform(500,2000),
}

# sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)  # , rule="c")
# sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=[1,1,1])#, rule="c")
# sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=3)#, rule="c")

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
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=["Ne"])
campaign.apply_analysis(analysis)
plot_grid_2D(0, "grid0.png")

i = 0
sobols_error = 1e6
error_vs_its = []
samples_vs_its = []
while sobols_error > 1e-4:
    print("Iteration "+str(i))
    i += 1
    refine_sampling_plan(1)
    campaign.apply_analysis(analysis)
    results = campaign.last_analysis
    samples_vs_its.append(sampler.n_samples)

    plot_grid_2D(i, "grid" + str(i) + ".png")
    moment_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "moments" + str(i) + ".png"
    )
    sobols_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "sobols_first" + str(i) + ".png"
    )
    sobols_second_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "sobols_second" + str(i) + ".png"
    )
    distribution_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "distribution" + str(i) + ".png"
    )
    plt.figure()
    results.plot_sobols_first(
        "Ne",
        ylabel="iteration" + str(i),
        xlabel=r"$\rho$",
        filename=sobols_plot_filename,
    )
    plt.ylim(0, 1)
    plt.savefig("sobols" + str(i) + ".png")
    plt.close()

    plt.figure()
    custom_moments_plot(results, moment_plot_filename, i)
    plt.close()

    # Prevent overwrite of old fig
    plt.figure("stat_conv").clear()
    analysis.plot_stat_convergence()
    plt.savefig("stat_convergence.png")
    plt.close()

    sobols = analysis.get_sobol_indices("Ne")

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

    if i > 100:
        break

    # print(results.sobols_second("T"))
    # plt.figure()
    # results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

# results = campaign.analyse(qoi_cols=["T"])


###plt.figure()
###sobols = []
#### retrieve the Sobol indices from the results object
###params = list(sampler.vary.get_keys())
###for param in params:
###    sobols.append(results._get_sobols_first('T', param))
#### make a bar chart
###print(sobols)
###fig = plt.figure()
###ax = fig.add_subplot(111, title='First-order Sobol indices')
###ax.bar(range(len(sobols)), height=np.array(sobols).flatten())
###ax.set_xticks(range(len(sobols)))
###ax.set_xticklabels(params)
###plt.xticks(rotation=90)
###plt.tight_layout()
###plt.savefig("sobols.png")


results.plot_sobols_first("Ne", xlabel=r"$\rho$", filename=sobols_plot_filename)

fig, ax = plt.subplots()
xvalues = np.arange(len(results.describe("Ne", "mean")))
ax.fill_between(
    xvalues,
    results.describe("Ne", "mean") - results.describe("Ne", "std"),
    results.describe("Ne", "mean") + results.describe("Ne", "std"),
    label="std",
    alpha=0.2,
)
ax.plot(xvalues, results.describe("Ne", "mean"), label="mean")
try:
    ax.plot(xvalues, results.describe("Ne", "1%"), "--", label="1%", color="black")
    ax.plot(xvalues, results.describe("Ne", "99%"), "--", label="99%", color="black")
except RuntimeError:
    pass
ax.grid(True)
ax.set_ylabel(r"$N_e$")
ax.set_xlabel(r"$\rho$")
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

print(f"Results are in:\n\t{moment_plot_filename}\n\t{sobols_plot_filename}")
