#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt


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
###        campaign.draw_samples()
###        campaign.populate_runs_dir()
###        campaign.apply_for_each_run_dir(
###            uq.actions.ExecuteLocal(cmd, interpret="python3")
###        )
###        campaign.collate()
        campaign.execute().collate()

        # accept one of the multi indices of the new admissible set
        data_frame = campaign.get_collation_result()
        analysis.adapt_dimension("T", data_frame)

def plot_grid_2D(i,filename="out.pdf"):
    fig = plt.figure(figsize=[12,4])
    ax1 = fig.add_subplot(121)#, xlim=[0.15, 4.05], ylim=[0.45, 1.55], xlabel='conduction:chi', ylabel='T:scale', title='(x1, x2) plane')
    ax2 = fig.add_subplot(122)#, xlim=[0.45, 1.55], ylim=[0.45*np.pi, 1.55*np.pi], xlabel='T:gauss_width', ylabel='T:gauss_centre', title='(x3, x4) plane')
    #ax3 = fig.add_subplot(133, xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel='x19', ylabel='x20', title='(x19, x20) plane')
    
    accepted_grid = sampler.generate_grid(analysis.l_norm)
    ax1.plot(accepted_grid[:,0], accepted_grid[:,1], 'o')
    ax2.plot(accepted_grid[:,2], accepted_grid[:,3], 'o')
    ax1.set_title("iteration "+str(i))
    #ax3.plot(accepted_grid[:,18], accepted_grid[:,19], 'o')
    
    plt.tight_layout()
    plt.savefig(filename)

def custom_moments_plot(results,filename,i):
    fig, ax = plt.subplots()
    xvalues = np.arange(len(results.describe("T", 'mean')))
    ax.fill_between(xvalues, results.describe("T", 'mean') -
                    results.describe("T", 'std'), results.describe("T", 'mean') +
                    results.describe("T", 'std'), label='std', alpha=0.2)
    ax.plot(xvalues, results.describe("T", 'mean'), label='mean')
    try:
        ax.plot(xvalues, results.describe("T", '1%'), '--', label='1%', color='black')
        ax.plot(xvalues, results.describe("T", '99%'), '--', label='99%', color='black')
    except RuntimeError:
        pass
    ax.grid(True)
    ax.set_ylabel("T")
    ax.set_xlabel(r"$\rho$")
    ax.set_title("iteration "+str(i))
    ax.legend()
    fig.savefig(filename)

encoder = boutvecma.BOUTEncoder(template_input="models/conduction/data/BOUT.inp")
# decoder = boutvecma.LogDataBOUTDecoder(variables=["T"])
decoder = boutvecma.SimpleBOUTDecoder(variables=["T"])
params = {
    "conduction:chi": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:scale": {"type": "float", "min": 0.0, "max": 1e3, "default": 1.0},
    "T:gauss_width": {"type": "float", "min": 0.0, "max": 1e3, "default": 0.2},
    "T:gauss_centre": {"type": "float", "min": 0.0, "max": 2 * np.pi, "default": np.pi},
}
actions = uq.actions.local_execute(
    encoder,
    os.path.abspath(
        "build/models/conduction/conduction -q -q -q -q  -d . |& tee run.log"
    ),
    decoder,
)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)
print(type(campaign))
# campaign.set_app("1D_conduction")

vary = {
    "conduction:chi": chaospy.Uniform(0.2, 4.0),
    # "conduction:chi": chaospy.LogUniform(0.2, 4.0),
    "T:scale": chaospy.Uniform(0.5, 1.5),
    "T:gauss_width": chaospy.Uniform(0.5, 1.5),
    "T:gauss_centre": chaospy.Uniform(0.5 * np.pi, 1.5 * np.pi),
}

# sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=3)  # , rule="c")
# sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=[1,1,1])#, rule="c")
# sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=3)#, rule="c")

sampler = uq.sampling.SCSampler(vary=vary, polynomial_order=1,
                                quadrature_rule="C",
                                sparse=True, growth=True,
                                midpoint_level1=True,
                                dimension_adaptive=True)
campaign.set_sampler(sampler)

print(f"Computing {sampler.n_samples} samples")

time_start = time.time()
campaign.execute().collate()

#results = campaign.analyse(qoi_cols=["T"])

# Create an analysis class and run the analysis.
print("here")
analysis = uq.analysis.SCAnalysis(sampler=sampler, qoi_cols=["T"])
print("here")
campaign.apply_analysis(analysis)
print(type(analysis))
print(type(campaign))
plot_grid_2D(0,"grid0.png")

for i in np.arange(1,100):
    refine_sampling_plan(1)
    campaign.apply_analysis(analysis)
    results = campaign.last_analysis
    plot_grid_2D(i,"grid"+str(i)+".png")
    moment_plot_filename = os.path.join(f"{campaign.campaign_dir}", "moments"+str(i)+".png")
    sobols_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_first"+str(i)+".png")
    sobols_second_plot_filename = os.path.join(f"{campaign.campaign_dir}", "sobols_second"+str(i)+".png")
    distribution_plot_filename = os.path.join(
        f"{campaign.campaign_dir}", "distribution"+str(i)+".png"
    )
    plt.figure()
    results.plot_sobols_first("T", ylabel="iteration"+str(i), xlabel=r"$\rho$", filename=sobols_plot_filename)
    plt.ylim(0,1)
    plt.savefig("sobols"+str(i)+".png")
    plt.figure()
    custom_moments_plot(results,moment_plot_filename,i)
    #print(results.sobols_second("T"))
    #plt.figure()
    #results.plot_moments("T", xlabel=r"$\rho$", filename=moment_plot_filename)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

#results = campaign.analyse(qoi_cols=["T"])


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


results.plot_sobols_first("T", xlabel=r"$\rho$", filename=sobols_plot_filename)

fig, ax = plt.subplots()
xvalues = np.arange(len(results.describe("T", 'mean')))
ax.fill_between(xvalues, results.describe("T", 'mean') -
                results.describe("T", 'std'), results.describe("T", 'mean') +
                results.describe("T", 'std'), label='std', alpha=0.2)
ax.plot(xvalues, results.describe("T", 'mean'), label='mean')
try:
    ax.plot(xvalues, results.describe("T", '1%'), '--', label='1%', color='black')
    ax.plot(xvalues, results.describe("T", '99%'), '--', label='99%', color='black')
except RuntimeError:
    pass
ax.grid(True)
ax.set_ylabel("T")
ax.set_xlabel(r"$\rho$")
ax.legend()
fig.savefig(moment_plot_filename)

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
