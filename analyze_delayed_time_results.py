import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import uproot
import numpy as np
from JetTimingStudy import get_graph_range_cut
import matplotlib.pyplot as plt

# -- File path
envdata_file =  "/home/submit/rozalena/LLP_Project_Data/hadd_LLPSkim_minituple_job0.root" 

# -- Tree name
tree_name = "PerJet_NoSel"

# -- Variables to plot on a log scale
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

# residuals more than max residual will be marked as high resid data points
max_residual = 2

delayed_jet_prediction_model = lgb.Booster(model_file='lightgbm_delayed_time_results/delayed_jet_time_predict.txt')
Y_pred = delayed_jet_prediction_model.predict(X_test)

t_true = Y_test.values.ravel() # makes a multi dimennsional array 1d
t_pred = Y_pred.ravel()

high_residual_mask = np.abs(t_true - t_pred) >= max_residual
low_residual_mask = [not item for item in high_residual_mask]

detailed_vars_to_plot = [
    ("perJet_Timeavg", True, -10, 20, 50),
    ("perJet_TDCavg", True, 0, 2, 30),
    ("perJet_TDCavg_energyWeight_log", True, 0, 1, 30),
    ("perJet_TDCnDelayed", True, 0, 10, 30),
    ("perJet_EnergyFrac_Depth1", True, 0, 1, 30),
    ("perJet_NeutralHadEFrac", True, 0, 1, 30),
    ("perJet_Pt", True, 0, 500, 40),
    ("perJet_Mass", True, 0, 100, 40),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", True, 0, 1, 30),
    ("perJet_PhoEFrac", True, 0, 1, 30),
    ("perJet_EleEFrac_log", True, 0, 0.7, 25),
    ("perJet_MuonEFrac_log", True, 0, 0.2, 25),
    ("perJet_S_phiphi", True, 0, 0.5, 40),
    ("perJet_S_etaeta", True, 0, 0.5, 40),
    ("perJet_S_etaphi", True, 0, 0.5, 40),
    ("perJet_Eta", True, -1.5, 1.5, 50),
    ("perJet_Tracks_dR", True, 0, 100, 50),
    ("perJet_Track0dR", True, 0, 0.4, 50),
    ("perJet_Track0dEta", True, -0.5, 0.5, 50),
    ("perJet_Track0dPhi", True, -0.5, 0.5, 50),
    ("perJet_Track1dR", True, 0, 0.5, 50),
    ("perJet_Track1dEta", True, -0.5, 0.5, 50),
    ("perJet_Track1dPhi", True, -0.5, 0.5, 50),
    ("perJet_Track2dR", True, 0, 0.5, 50),
    ("perJet_Track2dEta", True, -0.5, 0.5, 50),
    ("perJet_Track2dPhi", True, -0.5, 0.5, 50),
    ("perJet_Frac_Track0Pt", True, 0, 50, 50),
    ("perJet_Frac_Track1Pt", True, 0, 0.6, 50),
    ("perJet_Frac_Track2Pt", True, 0, 0.3, 50),
    ("perJet_Frac_LeadingRechitE", True, 0, 0.8, 50),
    ("perJet_Frac_SubLeadingRechitE", True, 0, 0.2, 50),
    ("perJet_Frac_SSubLeadingRechitE", True, 0, 0.2, 50),
    ("perJet_AllRechitE", True, 0, 700, 50)
]

normalize = True 

def make_residual_overlay_plot(var_name, modified_range=False, lower_bound=None, upper_bound=None, bins=50, normalize_to_one=False, output_prefix="plot", data_arrays=X_test):
    print("Running plotting function: make_overlay_plot() for " + var_name)
    plt.figure(figsize=(8, 6))

    # Base selections
    data_array_as_np = data_arrays[var_name].to_numpy()
    
    if modified_range:
        range_cut_mask = get_graph_range_cut(data_arrays, var_name, lower_bound, upper_bound)
        high_resid_data_vals = data_array_as_np[high_residual_mask & range_cut_mask]
        low_resid_data_vals = data_array_as_np[low_residual_mask & range_cut_mask]
    else:
        high_resid_data_vals = data_array_as_np[high_residual_mask]
        low_resid_data_vals = data_array_as_np[low_residual_mask]

    # Draw data
    hist_range = (lower_bound, upper_bound) if modified_range else None
    
    if hist_range:
        bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
    else:
        bin_edges = bins

    hist_kwargs = dict(bins=bin_edges, range=hist_range, histtype='step', linewidth=2)

    if normalize_to_one:
        high_resid_data_weights = np.ones_like(high_resid_data_vals) / len(high_resid_data_vals) if len(high_resid_data_vals) > 0 else None # basically doing 1/N for the histogram to normalize it
        low_resid_data_weights = np.ones_like(low_resid_data_vals) / len(low_resid_data_vals) if len(low_resid_data_vals) > 0 else None # basically doing 1/N for the histogram to normalize it

        plt.hist(high_resid_data_vals, weights=high_resid_data_weights, label="High Resid Data Vals", color="black", **hist_kwargs)
        plt.hist(low_resid_data_vals, weights=low_resid_data_weights, label="Low Resid Data Vals", color="red", **hist_kwargs)
    else:
        plt.hist(high_resid_data_vals, label="High Resid Data Vals", color="black", **hist_kwargs)
        plt.hist(low_resid_data_vals, label="Low Resid Data Vals", color="red", **hist_kwargs)

    plt.xlabel(var_name)
    plt.title(var_name)
    plt.ylabel("Normalized Fraction of Entries" if normalize_to_one else "Entries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if "log" in var_name:
        plt.gca().set_yscale('log')

    outname = f"{output_prefix}_{var_name}_normalized.png" if normalize_to_one else f"{output_prefix}_{var_name}.png"
    outname = f"delayed_time_data_with_high_resid/{outname}"
    plt.savefig(outname)
    plt.close()

# -- Make residual overlay plot 
for var, modify_range, lower_bound, upper_bound, bins  in detailed_vars_to_plot:
    make_residual_overlay_plot(var, modified_range=modify_range, lower_bound=lower_bound, upper_bound=upper_bound, bins=bins, normalize_to_one=normalize, output_prefix="overlay", data_arrays=X_test)