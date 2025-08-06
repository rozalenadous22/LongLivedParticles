# LongLivedParticles

JetTimingStudy.py: contains the plotting functions to understand and visualize the data collected on jets. 2d_histograms folder contains 2d histograms of perJet_Timeavg vs all other variables we are analyzing the jet travel time and delay on. all_vars_decay_r_constraints folder contains 1d histograms of each variable with decay r cuts based on different detector locations of CMS. all_vars_time_constraints folder contains 1d histograms of each variable with LLP travel time cuts based on observed general ranges of LLP travel times. 

lightgbm_model.py: This file uses Monte Carlo data to predict LLP travel times using the lightgbm ML model. Running the file will output residual, feature importance, and validation metric result (RMSE) plots in (folder). 

cms_data_predict.py: This file uses CMS particle collision data to predict jet delay times using the lightgbm ML model. Running the file will output residual, feature importance, and validation metric result (RMSE) plots in (folder). 
