import lightgbm as lgb
import pandas as pd
from sklearn.model_selection import train_test_split
import uproot
import numpy as np
#from JetTimingStudy import make_overlay_plot


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


# change based on what 
max_residual = 2


delayed_jet_prediction_model = lgb.Booster(model_file='delayed_jet_time_predict.txt')
Y_pred = delayed_jet_prediction_model.predict(X_test)

t_true = Y_test.values.ravel() # makes a multi dimennsional array 1d
t_pred = Y_pred.ravel()

residual_mask = np.abs(t_true - t_pred) >= max_residual


detailed_vars_to_plot = [
    ("perJet_Timeavg", True, 0, 2, 30),
    ("perJet_TDCavg", True, 0, 2, 30),
    ("perJet_TDCavg_energyWeight", True, 0, 2, 30),
    ("perJet_TDCnDelayed", True, 0, 10, 30),
    ("perJet_EnergyFrac_Depth1", True, 0, 1, 30),
    ("perJet_NeutralHadEFrac", True, 0, 1, 30),
    ("perJet_Pt", True, 0, 250, 40),
    ("perJet_Mass", True, 0, 40, 40),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", True, 0, 1, 30),
    ("perJet_PhoEFrac", True, 0, 1, 30),
    ("perJet_EleEFrac", True, 0, 0.2, 25),
    ("perJet_MuonEFrac", True, 0, 0.2, 25),
    ("perJet_S_phiphi", True, 0, 0.08, 40),
    ("perJet_S_etaeta", True, 0, 0.08, 40),
    ("perJet_S_etaphi", True, 0, 0.05, 40),
    ("perJet_Eta", False, 0, 1, 50),
    ("perJet_Tracks_dR", False, 0, 1, 50),
    ("perJet_Track0dR", False, 0, 1, 50),
    ("perJet_Track0dEta", False, 0, 1, 50),
    ("perJet_Track0dPhi", False, 0, 1, 50),
    ("perJet_Track1dR", False, 0, 1, 50),
    ("perJet_Track1dEta", False, 0, 1, 50),
    ("perJet_Track1dPhi", False, 0, 1, 50),
    ("perJet_Track2dR", False, 0, 1, 50),
    ("perJet_Track2dEta", False, 0, 1, 50),
    ("perJet_Track2dPhi", False, 0, 1, 50),
    ("perJet_Frac_Track0Pt", False, 0, 1, 50),
    ("perJet_Frac_Track1Pt", False, 0, 1, 50),
    ("perJet_Frac_Track2Pt", False, 0, 1, 50),
    ("perJet_Frac_LeadingRechitE", False, 0, 1, 50),
    ("perJet_Frac_SubLeadingRechitE", False, 0, 1, 50),
    ("perJet_Frac_SSubLeadingRechitE", False, 0, 1, 50),
    ("perJet_AllRechitE", False, 0, 1, 50)
]


print(len(detailed_vars_to_plot))

normalize = True 


# # outdated jettiming study... 
# for var, modify_range, lower_bound, upper_bound, bins  in detailed_vars_to_plot:
#     make_overlay_plot(var, modified_range=modify_range, lower_bound=lower_bound, upper_bound=upper_bound, bins=bins, extra_mc_cuts=None, normalize_to_one=normalize, output_prefix="overlay")


