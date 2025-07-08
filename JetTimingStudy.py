import uproot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# -- File paths
envdata_file =  "/home/submit/rozalena/LLP_Project_Data/Data_LLPskim_Run2023Cv4_ntuplesv3_20June.root" 
mc_file = "/home/submit/rozalena/LLP_Project_Data/MC_LLP_mh125_ms50_ctau3m_ntuplesv3_5June_small.root"

# -- Tree name
tree_name = "PerJet_NoSel"

# -- Variables to plot
variables_to_plot = ["perJet_TDCavg", "perJet_TDCavg_energyWeight", "perJet_TDCnDelayed", "perJet_Timeavg", "perJet_EnergyFrac_Depth1", "perJet_NeutralHadEFrac", "perJet_Pt", "perJet_Mass", "perJet_Area", "perJet_ChargedHadEFrac", "perJet_PhoEFrac", "perJet_EleEFrac", "perJet_MuonEFrac", "perJet_MatchedLLP_DecayZ", "perJet_MatchedLLP_DecayR", "perJet_MatchedLLP_TravelTime", "perJet_MatchedLLP_Eta", "perJet_S_phiphi", "perJet_S_etaeta", "perJet_S_etaphi"]

# -- Ideal for time constraints 
detailed_time_constraint_vars = [
    ("perJet_EnergyFrac_Depth1", True, 0, 1, 30),
    ("perJet_NeutralHadEFrac", True, 0, 1, 30),
    ("perJet_Pt", True, 0, 250, 40),
    ("perJet_Mass", True, 0, 40, 40),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", True, 0, 1, 30),
    ("perJet_PhoEFrac", True, 0, 1, 25),
    ("perJet_EleEFrac", True, -0.1, 0.2, 50),
    ("perJet_MuonEFrac", True, 0, 0.2, 50),
    ("perJet_MatchedLLP_DecayZ", True, -320, 320, 40),
    ("perJet_MatchedLLP_DecayR", True, 0, 300, 40),
    ("perJet_MatchedLLP_TravelTime", True, 0, 20, 25),
    ("perJet_MatchedLLP_Eta", True, -3, 3, 40),
    ("perJet_S_phiphi", True, 0, 0.08, 40),
    ("perJet_S_etaeta", True, 0, 0.08, 40),
    ("perJet_S_etaphi", True, 0, 0.05, 40),
    ("perJet_TDCavg", True, 0, 2, 40),
    ("perJet_TDCavg_energyWeight", True, 0, 2, 40),
    ("perJet_Timeavg", True, -5, 15, 30),
    ("perJet_TDCnDelayed", True, 0, 10, 30)
]
 
detailed_decay_r_constraint_vars = [
    ("perJet_EnergyFrac_Depth1", True, 0, 1, 30),
    ("perJet_NeutralHadEFrac", True, 0, 1, 30),
    ("perJet_Pt", True, 0, 250, 40),
    ("perJet_Mass", True, 0, 40, 40),
    ("perJet_Area", True, 0.3, 0.7, 25),
    ("perJet_ChargedHadEFrac", True, 0, 1, 30),
    ("perJet_PhoEFrac", True, 0, 1, 30),
    ("perJet_EleEFrac", True, 0, 0.2, 25),
    ("perJet_MuonEFrac", True, 0, 0.2, 25),
    ("perJet_MatchedLLP_DecayZ", True, -320, 320, 40),
    ("perJet_MatchedLLP_DecayR", True, 0, 300, 40),
    ("perJet_MatchedLLP_TravelTime", True, 0, 25, 20),
    ("perJet_MatchedLLP_Eta", True, -3, 3, 40),
    ("perJet_S_phiphi", True, 0, 0.08, 40),
    ("perJet_S_etaeta", True, 0, 0.08, 40),
    ("perJet_S_etaphi", True, 0, 0.05, 40),
    ("perJet_TDCavg", True, 0, 2, 30),
    ("perJet_TDCavg_energyWeight", True, 0, 2, 30),
    ("perJet_Timeavg", True, 0, 2, 30),
    ("perJet_TDCnDelayed", True, 0, 10, 30)
]

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
def make_overlay_plot(var_name, modified_range=False, lower_bound=None, upper_bound=None, bins=50, extra_mc_cuts=None, normalize_to_one=False, output_prefix="plot"):
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
        for label, (min, max), color, extra_mc_cuts_function in extra_mc_cuts:
            if modified_range:
                cut_mask = mc_mask & extra_mc_cuts_function(mc_arrays, min, max) & get_graph_range_cut(mc_arrays, var_name, lower_bound, upper_bound)
            else:
                cut_mask = mc_mask & extra_mc_cuts_function(mc_arrays, min, max)
            vals = mc_arrays[var_name][cut_mask]
            if normalize_to_one and len(vals) > 0:
                weights = np.ones_like(vals) / len(vals)
            else:
                weights = None
            if var_name == "perJet_EleEFrac":
                hist_vals, _ = np.histogram(vals, bins=bin_edges, weights=weights)
                print(f"{label} plotted bin values: {hist_vals}")
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
    outname = f"all_vars_decay_r_constraints/{outname}"
    plt.savefig(outname)
    plt.close()

    print("Saved plots to current directory!")

# def make_2d_hist_plot():
#     print("Running plotting function: make_2d_hist_plot()")
#     plt.figure(figsize=(8, 6))

#     # four decay r regions (cm): Tracker-inner, Tracker-outer, ECAL, HCAL
#     decayr_bins = np.array([0, 10, 129, 177, 295])     

#     # four timing regions (ns)  :  prompt 0-1, 1-5, 5-10, 10-20
#     time_bins   = np.array([0, 1, 5, 10, 20])  

#     decay_r_data = mc_arrays["perJet_MatchedLLP_DecayR"][mc_mask]
#     travel_time_data = mc_arrays["perJet_MatchedLLP_TravelTime"][mc_mask]

#     counts, _, _ = np.histogram2d(decay_r_data, travel_time_data,
#                                 bins=[decayr_bins, time_bins])
#     counts_norm = counts / counts.sum()

#     fig, ax = plt.subplots(figsize=(6, 6))
#     im = ax.imshow(counts_norm.T, origin='lower', cmap='viridis', aspect='equal')

#     ax.set_xticks(np.arange(4) + 0.5)
#     ax.set_yticks(np.arange(4) + 0.5)
#     x_range_labels = ["10", "129", "177", "295"]
#     ax.set_xticklabels(x_range_labels)   
#     y_range_labels = ["1", "5", "10", "20"]
#     ax.set_yticklabels(y_range_labels)

#     ax.set_xticks(np.arange(-.5, 4, 1), minor=True)
#     ax.set_yticks(np.arange(-.5, 4, 1), minor=True)
#     ax.grid(which='minor', color='w', linewidth=1)

#     ax.set_xlabel("LLP Decay-R  [cm]")
#     ax.set_ylabel("LLP Travel Time  [ns]")
#     ax.set_title("Normalised Fraction of Jets")
#     fig.colorbar(im, ax=ax, label="Fraction of entries")
#     plt.tight_layout()
#     plt.show()

#     plt.savefig("decay_r_vs_time/graph")
#     plt.close()
#     print("Saved plot to current directory!")

# def make_2d_hist():
#     print("Running plotting function: make_2d_hist()")
#     x_lo, x_hi = 0, 300      # Decay-R range  [cm]
#     y_lo, y_hi = 0, 20       # Travel-time    [ns]

#     n_xbins    = 30
#     n_ybins    = 20

#     decayr_edges = np.linspace(x_lo, x_hi, n_xbins + 1)
#     time_edges   = np.linspace(y_lo, y_hi, n_ybins + 1)

#     decay_r = mc_arrays["perJet_MatchedLLP_DecayR"][mc_mask]
#     travel_t = mc_arrays["perJet_MatchedLLP_TravelTime"][mc_mask]

#     counts, _, _ = np.histogram2d(decay_r, travel_t, bins=[decayr_edges, time_edges])
#     counts_norm  = counts / counts.sum()

   
#     fig, ax = plt.subplots(figsize=(7, 5))

#     pcm = ax.pcolormesh(decayr_edges, time_edges, counts_norm.T,
#                         cmap='viridis', shading='auto',
#                         norm=mpl.colors.LogNorm(vmin=1e-5, vmax=counts_norm.max()))
#     # log color scale so you can see when there are 0 jets

#     ax.set_xlabel("LLP Decay-R [cm]")
#     ax.set_ylabel("LLP Travel Time [ns]")
#     ax.set_title("Normalized Fraction of Jets (uniform bins)")

#     cbar = fig.colorbar(pcm, ax=ax)
#     cbar.set_label("Fraction of entries")

#     plt.tight_layout()
#     plt.show()

#     plt.savefig("2d_histograms/decay_r_and_travel_time")
#     plt.close()
#     print("Saved plot to current directory!")

def make_2d_no_cuts_hist_plot(x_array=None, y_array=None, x_array_mask=None, y_array_mask=None, x_data=None, y_data=None):
    x_lo, x_hi, n_xbins, x_var_name = x_data[0], x_data[1], x_data[2], x_data[3]
    y_lo, y_hi, n_ybins, y_var_name = y_data[0], y_data[1], y_data[2], y_data[3]

    print(f"Running plotting function: make_2d_no_cuts_hist_plot() for {x_var_name} and {y_var_name}")

    x_axis_edges = np.linspace(x_lo, x_hi, n_xbins + 1)
    y_axis_edges   = np.linspace(y_lo, y_hi, n_ybins + 1)

    x_axis_graph = x_array[x_var_name][x_array_mask] 
    y_axis_graph = y_array[y_var_name][y_array_mask]


    counts, _, _ = np.histogram2d(x_axis_graph, y_axis_graph, bins=[x_axis_edges, y_axis_edges])
    counts_norm  = counts / counts.sum()

   
    fig, ax = plt.subplots(figsize=(7, 5))

    pcm = ax.pcolormesh(x_axis_edges, y_axis_edges, counts_norm.T,
                        cmap='viridis', shading='auto',
                        norm=mpl.colors.LogNorm(vmin=1e-5, vmax=counts_norm.max()))
    # log color scale so you can see when there are 0 jets

    ax.set_xlabel(x_var_name)
    ax.set_ylabel(y_var_name)
    ax.set_title("Normalized Fraction of Jets (uniform bins)")

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("Fraction of entries")

    plt.tight_layout()
    plt.show()

    plt.savefig(f"2d_histograms/{x_var_name}_and_{y_var_name}")
    plt.close()
    print("Saved plot to current directory!")


# -- Extra MC regions to compare (optional)
extra_decay_r_mc_regions = [
    ("DecayR Tracker 0-10 cm", (0, 10), "green", get_decay_r_mc_cut),
    ("DecayR Tracker 10-129 cm", (10, 129), "red", get_decay_r_mc_cut),
    ("DecayR ECAL 129–177 cm", (129, 177), "orange", get_decay_r_mc_cut),
    ("DecayR HCAL 177-295 cm", (177, 295), "blue", get_decay_r_mc_cut),
    ("η tracker -1.4-1.4", (-1.4, 1.4), "purple", get_eta_mc_cut)
]

extra_time_mc_regions = [
    ("Time 0-1 ns", (0, 1), "green", get_timing_mc_cut),
    ("Time 1-5 ns", (1, 5), "red", get_timing_mc_cut),
    ("Time 5-10 ns", (5, 10), "orange", get_timing_mc_cut),
    ("Time 10-20 ns", (10, 20), "blue", get_timing_mc_cut),
    ("η tracker -1.4-1.4", (-1.4, 1.4), "purple", get_eta_mc_cut)
]

# -- Loop over variables and make plots with normalization option
normalize = True  # Set this flag to True or False based on your need

# Change based on type of constraint used (time or decay r)
constraint_type, detailed_variables_to_plot = extra_decay_r_mc_regions, detailed_decay_r_constraint_vars 

# for var, modify_range, lower_bound, upper_bound, bins  in detailed_variables_to_plot:
#     make_overlay_plot(var, modified_range=modify_range, lower_bound=lower_bound, upper_bound=upper_bound, bins=bins, extra_mc_cuts=constraint_type, normalize_to_one=normalize, output_prefix="overlay")

two_d_hist_info = [
    (0, 2, 30, "perJet_Timeavg"), 
    (0, 2, 30, "perJet_TDCavg_energyWeight"), 
    (0, 20, 20, "perJet_MatchedLLP_TravelTime"), 
    (0, 300, 30, "perJet_MatchedLLP_DecayR"), 
    ]
# time avg vs tdc avg energy weight
make_2d_no_cuts_hist_plot(x_array=data_arrays, y_array=data_arrays, x_array_mask=data_mask, y_array_mask=data_mask, x_data=two_d_hist_info[0], y_data=two_d_hist_info[1])
# time avg vs matched LLP travel time
make_2d_no_cuts_hist_plot(x_array=mc_arrays, y_array=mc_arrays, x_array_mask=mc_mask, y_array_mask=mc_mask, x_data=two_d_hist_info[0], y_data=two_d_hist_info[2])
# matched LLP decay R vs travel time
make_2d_no_cuts_hist_plot(x_array=mc_arrays, y_array=mc_arrays, x_array_mask=mc_mask, y_array_mask=mc_mask, x_data=two_d_hist_info[3], y_data=two_d_hist_info[2])