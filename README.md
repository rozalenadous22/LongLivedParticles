# LongLivedParticles

# To Run Code

 <ins>Set Up Environment for Running Script: </ins>

conda create –name JetTimingProject

conda activate JetTimingProject

pip install -r requirements.txt

<ins>To Run a File:</ins>

python3 {file_name.py} # plots will be generated and saved to current directory

conda deactivate

# Project Files and Folders

<b>NOTE: Files will need to be rerun on new data files, current plots are based on old data files and are incorrect.</b> 

<ins>JetTimingStudy.py:</ins> Contains the plotting functions to understand and visualize the data collected on jets. 

<ins>2d_histograms/</ins> folder contains 2d histograms of perJet_MatchedLLP_TravelTime vs all other variables we are analyzing the jet travel time and delay on. 

<ins>all_vars_decay_r_constraints/</ins> folder contains 1d histograms of each variable with decay r cuts based on different detector locations of CMS. 

<ins>all_vars_time_constraints/</ins> folder contains 1d histograms of each variable with LLP travel time cuts based on observed general ranges of LLP travel times. 

<ins>MC_traveltime_predict.py:</ins> This file uses Monte Carlo data to predict LLP travel times using the lightgbm ML model. Running the file will output residual, feature importance, and validation metric result (RMSE) plots into lightgbm_LLP_travel_time_results/. 

<ins>jet_delay_time_predict.py:</ins> This file uses CMS particle collision data to predict jet delay times using the lightgbm ML model. Running the file will output residual, feature importance, and validation metric result (RMSE) plots into lightgbm_delayed_time_results/.

<ins>analyze_delayed_time_results:</ins> Contains the plotting functions to visualize the difference between high resid and low resid data points from the predicted jet delay times lightgbm model. Plots will be saved in delayed_time_with_high_resid/.



