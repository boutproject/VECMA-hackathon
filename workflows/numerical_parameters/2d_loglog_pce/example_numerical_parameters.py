#!/usr/bin/env python3

import boutvecma
import easyvvuq as uq
import chaospy
import os
import numpy as np
import time
import matplotlib.pyplot as plt

# Do one run with atol=-15 to find value to use as offset in error decoder
encoder = boutvecma.BOUTExpEncoder(template_input="../../../models/conduction/data/BOUT.inp")
decoder = boutvecma.SimpleBOUTDecoder(variables=["T"])
params = {
    "solver:atol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
}
actions = uq.actions.local_execute(encoder, os.path.abspath("../../../build/models/conduction/conduction -d . -q -q -q -q |& tee run.log"), decoder)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)

vary = {
    "solver:atol": chaospy.Uniform(-16, -14),
}

pord=0
sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=pord)
campaign.set_sampler(sampler)

print(f"Initial code evaluation: {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)
time_end = time.time()

print(f"Finished initial step, took {time_end - time_start}")

df = campaign.get_collation_result()

# Offset value is df["T"][50], temperature at centre of domain

# Begin UQ loop:
encoder = boutvecma.BOUTExpEncoder(template_input="../../../models/conduction/data/BOUT.inp")
decoder = boutvecma.AbsLogErrorBOUTDecoder(variables=["T"], error_value=float(df["T"][50]))
params = {
    "solver:atol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
    "solver:rtol": {"type": "float", "min": -15, "max": 0.0, "default": -15},
}
actions = uq.actions.local_execute(encoder, os.path.abspath("../../../build/models/conduction/conduction -d . -q -q -q -q |& tee run.log"), decoder)
campaign = uq.Campaign(name="Conduction.", actions=actions, params=params)

vary = {
    "solver:atol": chaospy.Uniform(-15, 0),
    "solver:rtol": chaospy.Uniform(-15, 0),
}

pord=6
sampler = uq.sampling.PCESampler(vary=vary, polynomial_order=pord)
campaign.set_sampler(sampler)

campaign.draw_samples()
print(f"Code will be evaluated {sampler.n_samples} times")

time_start = time.time()
campaign.execute().collate(progress_bar=True)
time_end = time.time()

print(f"Finished, took {time_end - time_start}")

df = campaign.get_collation_result()

list_atol = df["solver:atol"].to_numpy()
list_rtol = df["solver:rtol"].to_numpy()
cps_atol = np.unique(df["solver:atol"].to_numpy())
cps_rtol = np.unique(df["solver:rtol"].to_numpy())
#code_evals = df["T"].to_numpy()

results = campaign.analyse(qoi_cols=["T"])

# This is a surrogate for the error
# Labelled T, but the decode is for the error.
s = results.surrogate()
n=100
f = np.zeros(shape=(n,n))
cp = np.zeros(shape=(len(cps_atol),len(cps_rtol)))
avec = np.linspace(-15,0,n)
rvec = np.linspace(-15,0,n)
for ai, a in enumerate(avec):
    for ri, r in enumerate(rvec):
        f[ai,ri] = s({'solver:atol' : a, 'solver:rtol' : r})["T"]
        #f[ai,ri] = s(a,r)

for ai, a in enumerate(cps_atol):
    for ri, r in enumerate(cps_rtol):
        cp[ai,ri] = s({'solver:atol' : a, 'solver:rtol' : r})["T"]
        #cp[ai,ri] = s(a,r)

np.save('f_order_'+str(pord),f)
np.save('cps_atol_order_'+str(pord),cps_atol)
np.save('cps_rtol_order_'+str(pord),cps_rtol)
np.save('list_atol_order_'+str(pord),list_atol)
np.save('list_rtol_order_'+str(pord),list_rtol)
np.save('cp_order_'+str(pord),cp)

import matplotlib
import matplotlib.colors as colors   # palette and co.
def rainbow_line(plt,g2,abscissa,colorvec,cb_title,axis=0):
    jet = cm = plt.get_cmap('jet') 
    cNorm  = colors.Normalize(vmin=colorvec[0], vmax=colorvec[-1])
    #cNorm  = colors.Normalize(vmin=-32, vmax=0)
    #cNorm  = colors.LogNorm(vmin=colorvec[0], vmax=colorvec[-1])
    scalarMap = matplotlib.cm.ScalarMappable(norm=cNorm, cmap=jet)
    for ic in np.arange(0,np.size(colorvec))[::-1]:
            colorVal = scalarMap.to_rgba(colorvec[ic])
            colorText = (
             'color: (%4.2f,%4.2f,%4.2f)'%(colorVal[0],colorVal[1],colorVal[2])
            )
            if axis==0:
                plt.plot(abscissa,g2[:,ic],color=colorVal)   
            else:
                plt.plot(abscissa,g2[ic,:],color=colorVal)   

    # fake up the array of the scalar mappable. Urgh...
    scalarMap._A = []
    cb = plt.colorbar(scalarMap)
    cb.set_label(cb_title, labelpad=-1)

###for ai, a in enumerate(code_evals):
###    sols[ai] = code_evals[ai]

plt.figure()
plt.contourf(avec,rvec,f)
for i, _ in enumerate(list_atol):
    plt.plot(list_rtol[i],list_atol[i],'k.')
#plt.plot(areal,np.log10(0.02*rvec),'k-',label="atol=0.02 rtol")
plt.xlabel("$\log_{10}(rtol)$")
plt.ylabel("$\log_{10}(atol)$")
plt.title("log Error")
plt.xlim(-15,0)
plt.ylim(-15,0)
#plt.legend()
plt.colorbar()
plt.savefig("contour_log_order_"+str(pord)+".png")

plt.figure()
rainbow_line(plt,f,avec,rvec,"rtol",axis=0)
plt.legend()
plt.xlabel("atol")
plt.ylabel("Error at final time, centre point")
plt.savefig("E_vs_atol_order_"+str(pord)+".png")
