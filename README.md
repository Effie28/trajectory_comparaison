# trajectory_comparaison

Library to generate trajectories from GPS, RTS and ICP data and compare them to a reference trajectory.

## To plot trajectories compared to a reference :

### Generate reconstructed_traj
python3 scripts/trajectory_generation.py -gps trajectories/gt_gps_traj1.csv -rts trajectories/gt_rts_traj1_timestamp_modified.csv -icp trajectories/icp_odom_traj1_timestamp_modified.csv -ps

### ICP / IMU / Ground Truth comparison w/ alignement
evo_traj tum output/icp_reconstructed_traj.csv trajectories/imu_odom_traj1_timestamp_modified.csv --ref output/gt_reconstructed_traj.csv -pv --align_origin --full_check


### ICP / Ground Truth comparison

evo_traj tum output/icp_reconstructed_traj.csv --ref output/gt_reconstructed_traj.csv 

evo_traj tum output/icp_reconstructed_traj.csv output/gt_reconstructed_traj.csv --plot_mode xy -p

-v : verbose
--full_check : verbose ++

evo_config set plot_axis_marker_scale 0 : disable quiver
evo_config set plot_axis_marker_scale 0.5 : enable quiver