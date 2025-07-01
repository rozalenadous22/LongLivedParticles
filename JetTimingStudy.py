import uproot
import numpy as np
import matplotlib.pyplot as plt

# -- File paths
envdata_file =  "/home/submit/rozalena/LLP_Project_Data/Data_LLPskim_Run2023Cv4_ntuplesv3_20June.root" 
mc_file = "/home/submit/rozalena/LLP_Project_Data/MC_LLP_mh125_ms50_ctau3m_ntuplesv3_5June_small.root"

# -- Tree name
tree_name = "PerJet_NoSel"

# -- Variables to plot
variables_to_plot = ["perJet_TDCavg", "perJet_TDCavg_energyWeight", "perJet_TDCnDelayed", "perJet_Timeavg", "perJet_EnergyFrac_Depth1", "perJet_NeutralHadEFrac", "perJet_Pt", "perJet_Mass", "perJet_Area", "perJet_ChargedHadEFrac", "perJet_PhoEFrac", "perJet_EleEFrac", "perJet_MuonEFrac", "perJet_MatchedLLP_DecayZ", "perJet_MatchedLLP_DecayR", "perJet_MatchedLLP_TravelTime", "perJet_MatchedLLP_Eta", "perJet_S_phiphi", "perJet_S_etaeta", "perJet_S_etaphi"]


# -- Ideal for eta constraints 
detailed_eta_variables = [
    ("perJet_EnergyFrac_Depth1", True, -0.25, 1, 30),
    ("perJet_NeutralHadEFrac", True, 0, 1, 30),
    ("perJet_Pt", True, 0, 200, 30),
    ("perJet_Mass", True, -5, 30, 35),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", True, -0.1, 1, 40),
    ("perJet_PhoEFrac", False, 0, 1, 25),
    ("perJet_EleEFrac", True, 0, 0.2, 50),
    ("perJet_MuonEFrac", True, 0, 0.2, 50),
    ("perJet_MatchedLLP_DecayZ", True, -550, 550, 50),
    ("perJet_MatchedLLP_DecayR", True, 0, 300, 40),
    ("perJet_MatchedLLP_TravelTime", True, 0, 30, 50),
    ("perJet_MatchedLLP_Eta", True, -3, 3, 45),
    ("perJet_S_phiphi", True, -0.2, 0.2, 50),
    ("perJet_S_etaeta", True, -0.2, 0.2, 50),
    ("perJet_S_etaphi", True, -0.2, 0.2, 50),
    ("perJet_TDCavg", False, 0, 0, 40),
    ("perJet_TDCavg_energyWeight", False, 0, 0, 40),
    ("perJet_Timeavg", False, 0, 0, 30),
    ("perJet_TDCnDelayed", False, 0, 0, 30)
]

# -- Ideal for time constraints 
detailed_time_constraint_vars = [
    ("perJet_EnergyFrac_Depth1", True, 0, 1, 30),
    ("perJet_NeutralHadEFrac", False, 0, 1, 40),
    ("perJet_Pt", True, 0, 250, 40),
    ("perJet_Mass", True, 0, 40, 40),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", False, -0.1, 1, 30),
    ("perJet_PhoEFrac", False, 0, 1, 25),
    ("perJet_EleEFrac", True, -0.1, 0.2, 50),
    ("perJet_MuonEFrac", True, 0, 0.2, 50),
    ("perJet_MatchedLLP_DecayZ", True, -320, 320, 40),
    ("perJet_MatchedLLP_DecayR", True, -10, 300, 40),
    ("perJet_MatchedLLP_TravelTime", False, 0, 30, 25),
    ("perJet_MatchedLLP_Eta", True, -3, 3, 40),
    ("perJet_S_phiphi", True, 0, 0.08, 40),
    ("perJet_S_etaeta", True, 0, 0.08, 40),
    ("perJet_S_etaphi", True, 0, 0.05, 40),
    ("perJet_TDCavg", False, 0, 0, 40),
    ("perJet_TDCavg_energyWeight", False, 0, 0, 40),
    ("perJet_Timeavg", False, 0, 0, 30),
    ("perJet_TDCnDelayed", False, 0, 0, 30)
]
 
detailed_decay_r_vars = [
    ("perJet_EnergyFrac_Depth1", True, 0, 1, 30),
    ("perJet_NeutralHadEFrac", False, 0, 1, 30),
    ("perJet_Pt", True, 0, 250, 40),
    ("perJet_Mass", True, 0, 40, 50),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", False, -0.1, 1, 40),
    ("perJet_PhoEFrac", False, 0, 1, 25),
    ("perJet_EleEFrac", True, -0.1, 0.2, 50),
    ("perJet_MuonEFrac", True, 0, 0.2, 50),
    ("perJet_MatchedLLP_DecayZ", True, -320, 320, 40),
    ("perJet_MatchedLLP_DecayR", True, -10, 300, 40),
    ("perJet_MatchedLLP_TravelTime", False, 0, 30, 20),
    ("perJet_MatchedLLP_Eta", True, -3, 3, 40),
    ("perJet_S_phiphi", True, 0, 0.08, 40),
    ("perJet_S_etaeta", True, 0, 0.08, 40),
    ("perJet_S_etaphi", True, 0, 0.05, 40),
    ("perJet_TDCavg", False, 0, 0, 30),
    ("perJet_TDCavg_energyWeight", False, 0, 0, 30),
    ("perJet_Timeavg", False, 0, 0, 20),
    ("perJet_TDCnDelayed", False, 0, 0, 20)
]

# -- Edit this depending on constraints used
detailed_variables_to_plot = detailed_eta_variables 

# -- Selection functions
def load_tree(file_path, tree_name):
    file = uproot.open(file_path)
    return file[tree_name]

def apply_selection(array_dict, selection_mask):
    return {var: array[selection_mask] for var, array in array_dict.items()}

# -- Load data
data_tree = load_tree(envdata_file, tree_name)
mc_tree = load_tree(mc_file, tree_name)

# -- Read variables and selection flags
data_arrays = data_tree.arrays(variables_to_plot + ["Pass_WPlusJets"], library="np")
mc_arrays = mc_tree.arrays(variables_to_plot + ["Pass_LLPMatched", "perJet_MatchedLLP_DecayR"], library="np")

# -- Basic selections
# Pass_WPlusJets when a W+ boson is produced along with some jets (spray of particles that comes from particle collisions)
data_mask = data_arrays["Pass_WPlusJets"] == 1
mc_mask = mc_arrays["Pass_LLPMatched"] == 1 # would also add must pass the LLP trigger selection

data_vars_to_ignore = ["perJet_MatchedLLP_Eta", "perJet_MatchedLLP_DecayZ", "perJet_MatchedLLP_DecayR", "perJet_MatchedLLP_TravelTime"]
data_vars_plot_log = ["perJet_EleEFrac", "perJet_MuonEFrac", "perJet_TDCavg_energyWeight"]

# -- Option: Additional MC cuts

def get_decay_r_mc_cut(mc_arrays, decayr_min, decayr_max):
    return (mc_arrays["perJet_MatchedLLP_DecayR"] > decayr_min) & (mc_arrays["perJet_MatchedLLP_DecayR"] <= decayr_max)

def get_timing_mc_cut(mc_arrays, time_min, time_max):
    return (mc_arrays["perJet_MatchedLLP_TravelTime"] > time_min) & (mc_arrays["perJet_MatchedLLP_TravelTime"] <= time_max)

def get_eta_mc_cut(mc_arrays, eta_min, eta_max):
    return (mc_arrays["perJet_MatchedLLP_Eta"] > eta_min) & (mc_arrays["perJet_MatchedLLP_Eta"] <= eta_max)

def get_graph_range_cut(data_array, var_name, lower_bound, upper_bound): 
    return (data_array[var_name] >= lower_bound) & (data_array[var_name] <= upper_bound)

# -- Plotting function with normalization option
def make_overlay_plot(var_name, modified_range=False, lower_bound=None, upper_bound=None, bins=50, extra_mc_cuts=None, extra_mc_cuts_function=None, normalize_to_one=False, output_prefix="plot"):
    print("Running plotting function: make_overlay_plot() for " + var_name)
    plt.figure(figsize=(8, 6))

    # Base selections
    if var_name in data_vars_to_ignore:
        data_vals = []
    else:
        if modified_range:
            range_cut_mask = get_graph_range_cut(data_arrays, var_name, lower_bound, upper_bound)
            data_vals = data_arrays[var_name][data_mask & range_cut_mask]
        else:
            data_vals = data_arrays[var_name][data_mask]

    # Draw data
    hist_range = (lower_bound, upper_bound) if modified_range else None
    
    if hist_range:
        bin_edges = np.linspace(hist_range[0], hist_range[1], bins + 1)
    else:
        bin_edges = bins

    hist_kwargs = dict(bins=bin_edges, range=hist_range, histtype='step', linewidth=2)

    if normalize_to_one:
        data_weights = np.ones_like(data_vals) / len(data_vals) if len(data_vals) > 0 else None # basically doing 1/N for the histogram to normalize it
        plt.hist(data_vals, weights=data_weights, label="Prompt Background (Data)", color="black", **hist_kwargs)
    else:
        plt.hist(data_vals, label="Prompt Background (Data)", color="black", **hist_kwargs)

    # Extra MC cuts (optional)
    if extra_mc_cuts:
        for label, (tmin, tmax), color in extra_mc_cuts:
            if modified_range:
                cut_mask = mc_mask & extra_mc_cuts_function(mc_arrays, tmin, tmax) & get_graph_range_cut(mc_arrays, var_name, lower_bound, upper_bound)
            else:
                cut_mask = mc_mask & extra_mc_cuts_function(mc_arrays, tmin, tmax)
            vals = mc_arrays[var_name][cut_mask]
            if normalize_to_one and len(vals) > 0:
                weights = np.ones_like(vals) / len(vals)
            else:
                weights = None
            plt.hist(vals, weights=weights, label=f"LLP MC ({label})", color=color, **hist_kwargs)

    plt.xlabel(var_name)
    plt.title(var_name)
    plt.ylabel("Normalized Fraction of Entries" if normalize_to_one else "Entries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    if var_name in data_vars_plot_log: 
        plt.gca().set_yscale('log')

    outname = f"{output_prefix}_{var_name}_normalized.png" if normalize_to_one else f"{output_prefix}_{var_name}.png"
    outname = f"all_vars_eta_constraints/{outname}"
    plt.savefig(outname)
    plt.close()

    print("Saved plots to current directory!")

# -- Extra MC regions to compare (optional)
extra_decay_r_mc_regions = [
    ("DecayR Tracker 0-10 cm", (0, 10), "green"),
    ("DecayR Tracker 10-129 cm", (10, 129), "red"),
    ("DecayR ECAL 129–177 cm", (129, 177), "orange"),
    ("DecayR HCAL 177-295 cm", (177, 295), "blue")
]

extra_time_constraints = [
    ("Time 0-1 ns", (0, 1), "green"),
    ("Time 1-5 ns", (1, 5), "red"),
    ("Time 5-10 ns", (5, 10), "orange"),
    ("Time 10-20 ns", (10, 20), "blue")
]

exta_eta_constraints = [
    ("η tracker -1.4-1.4", (-1.4, 1.4), "green"),
]

# -- Loop over variables and make plots with normalization option
normalize = True  # Set this flag to True or False based on your need
for var, modify_range, lower_bound, upper_bound, bins  in detailed_variables_to_plot:
    make_overlay_plot(var, modified_range=modify_range, lower_bound=lower_bound, upper_bound=upper_bound, bins=bins, extra_mc_cuts=exta_eta_constraints, extra_mc_cuts_function=get_eta_mc_cut, normalize_to_one=normalize, output_prefix="overlay")
    # need to make this more effficient, only change one thing here for constraints... maybe pass in the function