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
import matplotlib.colors as colors  # palette and co.


def rainbow_line(plt, g2, abscissa, colorvec, cb_title, axis=0):
    jet = cm = plt.get_cmap("jet")
    cNorm = colors.Normalize(vmin=colorvec[0], vmax=colorvec[-1])
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    for ic in np.arange(0, np.size(colorvec))[::-1]:
        colorVal = scalarMap.to_rgba(colorvec[ic])
        colorText = "color: (%4.2f,%4.2f,%4.2f)" % (
            colorVal[0],
            colorVal[1],
            colorVal[2],
        )
        if axis == 0:
            plt.plot(abscissa, g2[:, ic], color=colorVal)
        else:
            plt.plot(abscissa, g2[ic, :], color=colorVal)

    # fake up the array of the scalar mappable. Urgh...
    scalarMap._A = []
    cb = plt.colorbar(scalarMap)
    cb.set_label(cb_title, labelpad=-1)


# Do one run with atol=-15 to find value to use as offset in error decoder
encoder = boutvecma.BOUTExpEncoder(
    template_input="../../../models/conduction/data/BOUT.inp"
)
decoder = boutvecma.SimpleBOUTDecoder(variables=["T"])
params = {
    "solver:atol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        "../../../build/models/conduction/conduction -d . -q -q -q -q |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)

vary = {
    "solver:atol": chaospy.Uniform(-16, -14),
}

pord = 0
sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=pord)
campaign.set_sampler(sampler)

print(f"Initial code evaluation: {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)
time_end = time.time()

print(f"Finished initial step, took {time_end - time_start}")

df = campaign.get_collation_result()

# Offset value is df["T"][50], temperature at centre of domain


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
        analysis.adapt_dimension("T", data_frame)


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
    plt.close()


# Begin UQ loop:
encoder = boutvecma.BOUTExpEncoder(
    template_input="../../../models/conduction/data/BOUT.inp"
)
decoder = boutvecma.AbsLogErrorBOUTDecoder(
    variables=["T"], error_value=float(df["T"][50])
)

params = {
    "solver:atol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
    "solver:rtol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        "../../../build/models/conduction/conduction -q -q -q -q  -d . |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)

vary = {
    "solver:atol": chaospy.Uniform(-14, 0),
    "solver:rtol": chaospy.Uniform(-14, 0),
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

print(f"Code will be evaluated {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)

# Create an analysis class and run the analysis.
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=["T"])
campaign.apply_analysis(analysis)
# plot_grid_2D(0,"grid0.png")

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
    custom_moments_plot(results, moment_plot_filename, i)
    plt.close()

    # Prevent overwrite of old fig
    plt.figure("stat_conv").clear()
    analysis.plot_stat_convergence()
    plt.savefig("stat_convergence.png")
    plt.close()

    sobols = analysis.get_sobol_indices("T")

    if i > 1:
        sobols_error = 0
        count = 0
        for j in sobols:
            count += 1
            sobols_error += np.mean(abs(sobols[j] - sobols_last[j]))
        sobols_error = sobols_error / count
        error_vs_its.append(sobols_error)
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

    df = campaign.get_collation_result()

    list_atol = df["solver:atol"].to_numpy()
    list_rtol = df["solver:rtol"].to_numpy()
    cps_atol = np.unique(df["solver:atol"].to_numpy())
    cps_rtol = np.unique(df["solver:rtol"].to_numpy())
    # code_evals = df["T"].to_numpy()

    n = 100
    f = np.zeros(shape=(n, n))
    cp = np.zeros(shape=(len(cps_atol), len(cps_rtol)))
    # sols = np.zeros(shape=(len(code_evals)))
    avec = np.linspace(-14, 0, n)
    rvec = np.linspace(-14, 0, n)
    for ai, a in enumerate(avec):
        for ri, r in enumerate(rvec):
            f[ai, ri] = analysis.surrogate(qoi="T", x=np.array([a, r]))[0]

    plt.figure()
    plt.contourf(avec, rvec, f)
    for ia, _ in enumerate(list_atol):
        plt.plot(list_rtol[ia], list_atol[ia], "r.")
    # plt.plot(areal,np.log10(0.02*rvec),'k-',label="atol=0.02 rtol")
    plt.xlabel("$\log_{10}(rtol)$")
    plt.ylabel("$\log_{10}(atol)$")
    plt.title("log Error")
    # plt.xlim(-15,0)
    # plt.ylim(-15,0)
    # plt.legend()
    plt.colorbar()
    plt.savefig("contour_log_order_" + str(i) + ".png")
    plt.close()

    plt.figure()
    plt.pcolormesh(avec, rvec, f)
    for ia, _ in enumerate(list_atol):
        plt.plot(list_rtol[ia], list_atol[ia], "r.")
    # plt.plot(areal,np.log10(0.02*rvec),'k-',label="atol=0.02 rtol")
    plt.xlabel("$\log_{10}(rtol)$")
    plt.ylabel("$\log_{10}(atol)$")
    plt.title("log Error")
    # plt.xlim(-15,0)
    # plt.ylim(-15,0)
    # plt.legend()
    plt.colorbar()
    plt.savefig("pcolormesh_log_order_" + str(i) + ".png")
    plt.close()

    if i > 5:
        break

    np.save("f_order_" + str(i), f)
    np.save("cps_atol_order_" + str(i), cps_atol)
    np.save("cps_rtol_order_" + str(i), cps_rtol)
    np.save("list_atol_order_" + str(i), list_atol)
    np.save("list_rtol_order_" + str(i), list_rtol)
    np.save("cp_order_" + str(i), cp)

time_end = time.time()

print(f"Finished, took {time_end - time_start}")


plt.figure()
rainbow_line(plt, f, avec, rvec, "rtol", axis=0)
plt.legend()
plt.xlabel("atol")
plt.ylabel("Error at final time, centre point")
plt.savefig("E_vs_atol_order_" + str(i) + ".png")
