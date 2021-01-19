#! /usr/bin/env python
"""
Run an EasyVVUQ campaign to analyze the sensitivity of the temperature
profile predicted by a simplified model of heat conduction in a
tokamak plasma.

This is done with PCE.
"""
import os
import easyvvuq as uq
import chaospy as cp
import pickle
import time
import numpy as np 
import matplotlib.pylab as plt
from boutvecma.encoder import BOUTEncoder

time_start = time.time()
# Set up a fresh campaign called "bout_pce."
my_campaign = uq.Campaign(name='bout_pce.')

# Define parameter space
params = {
    "conduction:chi":   {"type": "float",   "min": 0.1, "max": 2.0, "default": 1.0}, 
    "T:scale":       {"type": "float",   "min": 0.1,  "max": 2.0,    "default": 1.0}, 
    "timestep":       {"type": "float",   "min": 1e-3,  "max": 1e3,    "default": 100},
    "out_file": {"type": "string",  "default": "output.csv"}
}
###""" code snippet for writing the template file
str = ""
first = True
for k in params.keys():
    if first:
        str += '{"%s": "$%s"' % (k,k) ; first = False
    else:
        str += ', "%s": "$%s"' % (k,k)
str += '}'
print(str, file=open('bout.template','w'))
###"""

# Create an encoder, decoder and collater for PCE test app
encoder = BOUTEncoder("models/conduction/data/BOUT.inp")

decoder = uq.decoders.SimpleCSV(target_filename="output.csv",
                                output_columns=["te", "ne", "rho", "rho_norm"])

# Add the app (automatically set as current app)
my_campaign.add_app(name="bout_pce",
                    params=params,
                    encoder=encoder,
                    decoder=decoder)

time_end = time.time()
print('Time for phase 1 = %.3f' % (time_end-time_start))
time_start = time.time()

# Create the sampler
vary = {
    "conduction:chi":   cp.Uniform(0.1, 2.0),
    "T:scale":   cp.Uniform(0.1, 2.0)#,
}

# Associate a sampler with the campaign
my_campaign.set_sampler(uq.sampling.PCESampler(vary=vary, polynomial_order=3))

# Will draw all (of the finite set of samples)
my_campaign.draw_samples()
print('Number of samples = %s' % my_campaign.get_active_sampler().count)

time_end = time.time()
print('Time for phase 2 = %.3f' % (time_end-time_start))
time_start = time.time()

# Create and populate the run directories
my_campaign.populate_runs_dir()

time_end = time.time()
print('Time for phase 3 = %.3f' % (time_end-time_start))
time_start = time.time()

# Run the cases
cwd = os.getcwd().replace(' ', '\ ')      # deal with ' ' in the path
#cmd = f"{cwd}/run_conduction.py"
cmd = f"build/models/conduction/conduction -d ."
my_campaign.apply_for_each_run_dir(uq.actions.ExecuteLocal(cmd))#, interpret='python3'))

time_end = time.time()
print('Time for phase 4 = %.3f' % (time_end-time_start))
time_start = time.time()

# Collate the results
my_campaign.collate()

time_end = time.time()
print('Time for phase 5 = %.3f' % (time_end-time_start))
time_start = time.time()

# Post-processing analysis
my_campaign.apply_analysis(uq.analysis.PCEAnalysis(sampler=my_campaign.get_active_sampler(), qoi_cols=["te", "ne", "rho", "rho_norm"]))

time_end = time.time()
print('Time for phase 6 = %.3f' % (time_end-time_start))
time_start = time.time()

# Get Descriptive Statistics
results = my_campaign.get_last_analysis()
stats = results['statistical_moments']['te']
per = results['percentiles']['te']
sobols = results['sobols_first']['te']
rho = results['statistical_moments']['rho']['mean']
rho_norm = results['statistical_moments']['rho_norm']['mean']

time_end = time.time()
print('Time for phase 7 = %.3f' % (time_end-time_start))
time_start = time.time()

my_campaign.save_state("campaign_state.json")

###old_campaign = uq.Campaign(state_file="campaign_state.json", work_dir=".")

pickle.dump(results, open('fusion_results.pickle','bw'))
###saved_results = pickle.load(open('fusion_results.pickle','br'))

time_end = time.time()
print('Time for phase 8 = %.3f' % (time_end-time_start))

plt.ion()

# plot the calculated Te: mean, with std deviation, 10 and 90% and range
plt.figure() 
plt.plot(rho, stats['mean'], 'b-', label='Mean')
plt.plot(rho, stats['mean']-stats['std'], 'b--', label='+1 std deviation')
plt.plot(rho, stats['mean']+stats['std'], 'b--')
plt.fill_between(rho, stats['mean']-stats['std'], stats['mean']+stats['std'], color='b', alpha=0.2)
plt.plot(rho, per['p10'].ravel(), 'b:', label='10 and 90 percentiles')
plt.plot(rho, per['p90'].ravel(), 'b:')
plt.fill_between(rho, per['p10'].ravel(), per['p90'].ravel(), color='b', alpha=0.1)
plt.fill_between(rho, [r.lower[0] for r in results['output_distributions']['te']], [r.upper[0] for r in results['output_distributions']['te']], color='b', alpha=0.05)
plt.legend(loc=0)
plt.xlabel('rho [m]')
plt.ylabel('Te [eV]')
plt.title(my_campaign.campaign_dir)
plt.savefig('Te.png')

# plot the first Sobol results
plt.figure() 
for k in sobols.keys(): plt.plot(rho, sobols[k][0], label=k)
plt.legend(loc=0)
plt.xlabel('rho [m]')
plt.ylabel('sobols_first')
plt.title(my_campaign.campaign_dir)
plt.savefig('sobols_first.png')

# plot the total Sobol results
plt.figure() 
for k in results['sobols_total']['te'].keys(): plt.plot(rho, results['sobols_total']['te'][k][0], label=k)
plt.legend(loc=0)    
plt.xlabel('rho [m]')
plt.ylabel('sobols_total')
plt.title(my_campaign.campaign_dir)
plt.savefig('sobols_total.png')

# plot the distributions
plt.figure()
for i, D in enumerate(results['output_distributions']['te']):
    _Te = np.linspace(D.lower[0], D.upper[0], 101)
    _DF = D.pdf(_Te)
    plt.loglog(_Te, _DF, 'b-')
    plt.loglog(stats['mean'][i], np.interp(stats['mean'][i], _Te, _DF), 'bo')
    plt.loglog(stats['mean'][i]-stats['std'][i], np.interp(stats['mean'][i]-stats['std'][i], _Te, _DF), 'b*')
    plt.loglog(stats['mean'][i]+stats['std'][i], np.interp(stats['mean'][i]+stats['std'][i], _Te, _DF), 'b*')
    plt.loglog(per['p10'].ravel()[i],  np.interp(per['p10'].ravel()[i], _Te, _DF), 'b+')
    plt.loglog(per['p90'].ravel()[i],  np.interp(per['p90'].ravel()[i], _Te, _DF), 'b+')
plt.xlabel('Te')
plt.ylabel('distribution function')
plt.savefig('distribution_functions.png')

