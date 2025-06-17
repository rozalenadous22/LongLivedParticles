import uproot
import numpy as np
import matplotlib.pyplot as plt

# -- File paths
envdata_file =  "/home/submit/rozalena/LLP_Project_Data/Data_LLPskim_Run2023Cv1_ntuplesv3_5June.root"
mc_file = "/home/submit/rozalena/LLP_Project_Data/MC_LLP_mh125_ms50_ctau3m_ntuplesv3_5June_small.root"

# -- Tree name
tree_name = "PerJet_NoSel"

# -- Variables to plot
variables_to_plot = ["perJet_EnergyFrac_Depth1", "perJet_NeutralHadEFrac", "perJet_Pt", "perJet_Mass", "perJet_Area", "perJet_ChargedHadEFrac", "perJet_PhoEFrac", "perJet_EleEFrac", "perJet_MuonEFrac", "perJet_MatchedLLP_DecayZ", "perJet_MatchedLLP_DecayR", "perJet_MatchedLLP_TravelTime", "perJet_MatchedLLP_Eta", "perJet_S_phiphi", "perJet_S_etaeta", "perJet_S_etaphi"]
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
data_mask = data_arrays["Pass_WPlusJets"] == 1 # .sum() tells us only 2/79724 entries pass the data mask
mc_mask = mc_arrays["Pass_LLPMatched"] == 1 # would also add must pass the LLP trigger selection

# -- Option: Additional MC cuts
def get_decay_r_mc_cut(mc_arrays, decayr_min, decayr_max):
    return (mc_arrays["perJet_MatchedLLP_DecayR"] > decayr_min) & (mc_arrays["perJet_MatchedLLP_DecayR"] < decayr_max)

def get_timing_mc_cut(mc_arrays, time_min, time_max):
    return (mc_arrays["perJet_MatchedLLP_TravelTime"] > time_min) & (mc_arrays["perJet_MatchedLLP_TravelTime"] < time_max) # seems to be only time var

# -- Plotting function with normalization option
def make_overlay_plot(var_name, bins=50, range=None, extra_mc_cuts=None, normalize_to_one=False, output_prefix="plot"):
    print("Running plotting function: make_overlay_plot() for " + var_name)
    plt.figure(figsize=(8, 6))

    # Base selections
    data_vals = data_arrays[var_name][data_mask]
    mc_vals = mc_arrays[var_name][mc_mask]

    # Draw data
    hist_kwargs = dict(bins=bins, range=range, histtype='step', linewidth=2)

    if normalize_to_one:
        data_weights = np.ones_like(data_vals) / len(data_vals) if len(data_vals) > 0 else None # basically doing 1/N for the histogram to normalize it
        mc_weights = np.ones_like(mc_vals) / len(mc_vals) if len(mc_vals) > 0 else None
        plt.hist(data_vals, weights=data_weights, label="Data", color="black", **hist_kwargs)
        plt.hist(mc_vals, weights=mc_weights, label="MC (All)", color="blue", **hist_kwargs)
    else:
        plt.hist(data_vals, label="Data", color="black", **hist_kwargs)
        plt.hist(mc_vals, label="MC (All)", color="blue", **hist_kwargs)

    # Extra MC cuts (optional)
    if extra_mc_cuts:
        for label, (tmin, tmax), color in extra_mc_cuts:
            # cut_mask = mc_mask & get_decay_r_mc_cut(mc_arrays, rmin, rmax)
            cut_mask = mc_mask & get_timing_mc_cut(mc_arrays, tmin, tmax) # need to make the code cleaner so you can switch btwn diff cut types
            vals = mc_arrays[var_name][cut_mask]
            if normalize_to_one and len(vals) > 0:
                weights = np.ones_like(vals) / len(vals)
            else:
                weights = None
            plt.hist(vals, weights=weights, label=f"MC ({label})", color=color, **hist_kwargs)

    plt.xlabel(var_name)
    plt.ylabel("Normalized Fraction of Entries" if normalize_to_one else "Entries")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    outname = f"{output_prefix}_{var_name}_normalized.png" if normalize_to_one else f"{output_prefix}_{var_name}.png"
    outname = f"time_constraint_graphs/{outname}"
    plt.savefig(outname)
    plt.close()

    print("Saved plots to current directory!")

# -- Extra MC regions to compare (optional)
extra_decay_r_mc_regions = [
    ("DecayR 0-129 cm", (0, 129), "green"),
    ("DecayR 183-295 cm", (183, 295), "red")
]

more_decay_r_mc_regions_constraints = [
    ("DecayR Tracker 0-10 cm", (0, 10), "green"),
    ("DecayR Tracker 10-129 cm", (10, 129), "red"),
    ("DecayR ECAL 129–177 cm", (129, 177), "orange"),
    ("DecayR HCAL 177-295 cm", (177, 295), "purple")
]

extra_time_constraints = [
    ("Time 0-0.5 µs", (0, 0.5), "green"),
    ("Time 0.5-1 µs", (0.5, 1), "red"),
    ("Time 1-2 µs", (1, 2), "purple"),
]

# -- Loop over variables and make plots with normalization option
normalize = True  # Set this flag to True or False based on your need
for var in variables_to_plot:
    make_overlay_plot(var, bins=50, range=None, extra_mc_cuts=extra_time_constraints, normalize_to_one=normalize, output_prefix="overlay")