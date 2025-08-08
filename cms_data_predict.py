import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, classification_report

import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec


# -- File paths
envdata_file =  "/home/submit/rozalena/LLP_Project_Data/hadd_LLPSkim_minituple_job0.root" 

# -- Tree name
tree_name = "PerJet_NoSel"

data_vars_plot_log = ["perJet_EleEFrac", "perJet_MuonEFrac", "perJet_TDCavg_energyWeight"]

# -- Variables to plot
x_var_names = ["perJet_Timeavg", "perJet_TDCavg", "perJet_TDCavg_energyWeight", 
               "perJet_TDCnDelayed", "perJet_EnergyFrac_Depth1", "perJet_NeutralHadEFrac", 
               "perJet_Pt", "perJet_Mass", "perJet_Area", "perJet_ChargedHadEFrac", 
               "perJet_PhoEFrac", "perJet_EleEFrac", "perJet_MuonEFrac", "perJet_S_phiphi", 
               "perJet_S_etaeta", "perJet_S_etaphi", 'perJet_Eta','perJet_Tracks_dR', 'perJet_Track0dR', 
               'perJet_Track0dEta', 'perJet_Track0dPhi', 'perJet_Track1dR', 'perJet_Track1dEta', 
               'perJet_Track1dPhi','perJet_Track2dR', 'perJet_Track2dEta', 'perJet_Track2dPhi',
               'perJet_Frac_Track0Pt', 'perJet_Frac_Track1Pt', 'perJet_Frac_Track2Pt', 
               'perJet_Frac_LeadingRechitE', 'perJet_Frac_SubLeadingRechitE', 'perJet_Frac_SSubLeadingRechitE', 'perJet_AllRechitE']

# conormalize the variables: var = var -mean/rmse
y_var_names = ["QIE_phase"]

# -- Selection functions
def load_tree(file_path, tree_name):
    file = uproot.open(file_path)
    return file[tree_name]

data_tree = load_tree(envdata_file, tree_name)


x_data_arrays = data_tree.arrays(x_var_names , library="np") 
y_data_arrays = data_tree.arrays(y_var_names, library="np")

data_mask = (y_data_arrays["QIE_phase"] >= -10) & (y_data_arrays["QIE_phase"] <= 10)


X = pd.DataFrame()
Y = pd.DataFrame()


for var_name in x_var_names:
    if var_name in data_vars_plot_log:
        X[var_name+"_log"] = pd.DataFrame(np.log1p(x_data_arrays[var_name][data_mask]))
    else:
        X[var_name] = pd.DataFrame(x_data_arrays[var_name][data_mask])


Y["QIE_phase"] = pd.DataFrame(y_data_arrays["QIE_phase"][data_mask])


# random state = any integer so the results can be recreated everytime we run this program
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

train_data = lgb.Dataset(X_train, label=Y_train)
test_data = lgb.Dataset(X_test, label=Y_test)


params = { 
    'objective': 'regression_l1', # applys a pentaly to the loss function
    'boosting_type': 'gbdt', # standard, use gradient boosted method
    'num_leaves': 192, # higher = more complex calculations
    'min_data_in_leaf':50,
    'learning_rate': 0.05, # takes 5% of the previous tree's data for the next
    'metric': 'rmse', # root mean squared error to evaulate accuracy for next boosting round 
    'verbose': 3 # val > 1 will debug, 1 for just info
}

# total num of boosting rounds (trees) to train
num_round = 1000

evals_result = {}

bst = lgb.train(
    params,
    train_data,
    num_boost_round=num_round,
    valid_sets=[test_data],
    valid_names=["validation"],
    callbacks=[lgb.early_stopping(stopping_rounds=15), lgb.record_evaluation(evals_result)]
)

Y_pred = bst.predict(X_test)

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R-squared: {r2}")


bst.save_model("delayed_jet_time_predict.txt")


def make_calibration_plot(t_true, t_pred, # actual and model-predicted travel times [ns]
                          figname="delayed_travel_time_resid_plots.png",                 
                          n_hex=80, # resolution of hexbin grid (top)             
                          n_bins=30): # num bins for resid panel (bottom)              
    

    residual = t_pred - t_true

    lo, hi = np.percentile(np.r_[t_true, t_pred], [1, 99]) # 1st and 99th percentiles of both arrays ignores extreme outliers
    pad    = 0.05 * (hi - lo)
    lims   = (lo - pad, hi + pad)


    fig = plt.figure(figsize=(6, 7), constrained_layout=True)
    gs  = GridSpec(2, 1, height_ratios=[3, 1], figure=fig)
    axT = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1], sharex=axT)


    H, xedges, yedges = np.histogram2d(
    t_true, t_pred,
    bins=n_hex,
    range=[lims, lims]
    )

    pcm = axT.pcolormesh(
        xedges, yedges, H.T,
        cmap='viridis',
        norm=LogNorm(vmin=1),
        shading='auto'
    )

    cbar = fig.colorbar(pcm, ax=axT)
    cbar.set_label('events (log)')


    axT.plot(lims, lims, 'r:', lw=1, label='y = x')
    axT.set_xlim(lims);  axT.set_ylim(lims)
    axT.set_aspect('equal', adjustable='box') 
    axT.set_ylabel(r'$t_{\mathrm{pred}}\;\mathrm{[ns]}$')
    axT.legend(loc='upper left', fontsize=8)
    

    edges = np.linspace(*lims, n_bins + 1)
    centres = 0.5 * (edges[1:] + edges[:-1])
    digit = np.digitize(t_true, edges) - 1 # indices of the bins to which each value in an input array belongs

    mean = np.full(n_bins, np.nan)
    sig  = np.full(n_bins, np.nan)

    for i in range(n_bins):
        m = digit == i 
        if m.any():
            mean[i] = residual[m].mean()
            sig[i]  = residual[m].std()

    axB.errorbar(
            centres, mean, yerr=sig,
            fmt='o', ms=3.5, mfc='mediumpurple',
            mec='purple', ecolor='plum', capsize=2,
            label=r'mean $\pm1\sigma$'
    )
    axB.axhline(0, ls='--', c='red', label='y=x')
    axB.set_xlabel(r'$t_{\mathrm{true}}\;\mathrm{[ns]}$')
    axB.set_ylabel(r'$t_{\rm pred}-t_{\rm true}$')
    axB.legend(fontsize=8)

    plt.setp(axT.get_xticklabels(), visible=False)
    plt.title("jet delay time residual")
    fig.savefig(figname, dpi=300)
    plt.close(fig)


# plot the variable importance based on number of times the tree was split on it
lgb.plot_importance(bst, importance_type="split", max_num_features=10)
plt.xlabel("# times split based on var")
plt.ylabel("var name")
plt.tight_layout()
plt.savefig("feature_importance")
plt.close()


# plot the RMSE at every boosting round and label the final round 
final_round = bst.best_iteration
rmse_trace = evals_result["validation"]["rmse"]
rounds = np.arange(1, len(rmse_trace) + 1)

plt.plot(rounds, rmse_trace)
plt.xlabel("boosting round")
plt.ylabel("RMSE [ns]")
plt.axvline(x=final_round, linestyle='--', color='red', lw=1, label=f'best iter, x={final_round}')
plt.title("Validation RMSE vs. boosting round", pad=10)
plt.legend()
plt.tight_layout()
plt.savefig("delayed_time_RMSE_results")
plt.close()



t_true = Y_test.values.ravel() # makes a multi dimennsional array 1d
t_pred = Y_pred.ravel()
make_calibration_plot(t_true, t_pred)

# need to take the residual, if its more than 50% off 
