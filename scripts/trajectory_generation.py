# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
BASE_PATH = Path(__file__).absolute().parents[1] 
import point_to_point as ptp

# %%
df_rts1 = pd.read_csv(BASE_PATH / "traj1/traj1_groundtruth_rts_lidar.csv", names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
df_gps1 = pd.read_csv(BASE_PATH / "traj1/traj1_groundtruth_gps_lidar.csv", delimiter= ',')

df_rts2 = pd.read_csv(BASE_PATH / "traj2/traj2_groundtruth_rts_lidar.csv", names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
df_gps2 = pd.read_csv(BASE_PATH / "traj2/traj2_groundtruth_gps_lidar.csv", delimiter= ',')

df_rts1['Timestamp'] = pd.to_datetime(df_rts1['Timestamp'], unit='s') # Convert to datetime
print(df_rts1['Timestamp'].tail())
df_rts1['Timestamp'] -= pd.Timedelta(hours=4, seconds=6) # Subtract 4 hours from the RTS Timestamp column
df_rts1['Timestamp'] = df_rts1["Timestamp"].values.astype(np.int64) // (1*10 ** 9) # Convert to Unix Timestamp

df_rts2['Timestamp'] = pd.to_datetime(df_rts2['Timestamp'], unit='s') # Convert to datetime
print(df_rts2['Timestamp'].head())
df_rts2['Timestamp'] -= pd.Timedelta(hours=4, seconds=6) # Subtract 4 hours from the RTS Timestamp column
df_rts2['Timestamp'] = df_rts2["Timestamp"].values.astype(np.int64) // (1*10 ** 9) # Convert to Unix Timestamp


# %%
merged_data = pd.merge(
    df_rts1[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
    df_gps1[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw','PDOP','Sats']],
    on=['Timestamp'],
    how='outer',
    indicator=True
)
common_traj1 = merged_data[merged_data['_merge'] == 'both']
gps_only_traj1 = merged_data[merged_data['_merge'] == 'right_only']
rts_only_traj1 = merged_data[merged_data['_merge'] == 'left_only']

common_traj1.rename(
    columns={'X_x':'X_rts', 'Y_x':'Y_rts', 'Z_x':'Z_rts', 'X_y':'X_gps', 'Y_y':'Y_gps', 'Z_y':'Z_gps',
             'qx_x' : 'qx_rts', 'qy_x' : 'qy_rts', 'qz_x' : 'qz_rts', 'qw_x' : 'qw_rts'},
    inplace=True
)
gps_only_traj1.rename(
    columns={'X_y':'X', 'Y_y':'Y', 'Z_y':'Z', 'qx_y' : 'qx', 'qy_y' : 'qy', 'qz_y' : 'qz', 'qw_y' : 'qw'},
    inplace=True
)
rts_only_traj1.rename(
    columns={'X_x':'X', 'Y_x':'Y', 'Z_x':'Z', 'qx_x' : 'qx', 'qy_x' : 'qy', 'qz_x' : 'qz', 'qw_x' : 'qw'},
      inplace=True
)

gps_only_traj1.drop(['X_x', 'Y_x', 'Z_x', 'qx_x', 'qy_x', 'qz_x', 'qw_x', '_merge'], axis=1, inplace=True)
rts_only_traj1.drop(['X_y', 'Y_y', 'Z_y', 'qx_y', 'qy_y', 'qz_y', 'qw_y', 'PDOP', 'Sats', '_merge'], axis=1, inplace=True)

#Transform the common trajectory to the GPS frame
P_common = np.array([common_traj1['X_rts'], common_traj1['Y_rts'], common_traj1['Z_rts'], np.ones(len(common_traj1['X_rts']))])
Q_common = np.array([common_traj1['X_gps'], common_traj1['Y_gps'], common_traj1['Z_gps'], np.ones(len(common_traj1['X_gps']))])
T = ptp.minimization(P_common, Q_common)
P_common_transformed = T @ P_common
common_traj1['X_rts'] = P_common_transformed[0,:]
common_traj1['Y_rts'] = P_common_transformed[1,:]
common_traj1['Z_rts'] = P_common_transformed[2,:]

#Transform the RTS trajectory to the GPS frame
P_rts = np.array([df_rts1['X'], df_rts1['Y'], df_rts1['Z'],np.ones(len(df_rts1['X']))])
P_rts_transformed = T @ P_rts
df_rts1['X'] = P_rts_transformed[0,:]
df_rts1['Y'] = P_rts_transformed[1,:]
df_rts1['Z'] = P_rts_transformed[2,:]

#Transform the RTS only trajectory to the GPS frame
P_rts_only_traj1 = np.array([rts_only_traj1['X'], rts_only_traj1['Y'], rts_only_traj1['Z'], np.ones(len(rts_only_traj1['X']))])
P_rts_only_traj1_transformed = T @ P_rts_only_traj1
rts_only_traj1['X'] = P_rts_only_traj1_transformed[0,:]
rts_only_traj1['Y'] = P_rts_only_traj1_transformed[1,:]
rts_only_traj1['Z'] = P_rts_only_traj1_transformed[2,:]

#Concat the common, gps only and rts only trajectories
common_traj1.rename(
    columns={'X_rts':'X', 'Y_rts':'Y', 'Z_rts':'Z', 'qx_rts' : 'qx', 'qy_rts' : 'qy', 'qz_rts' : 'qz', 'qw_rts' : 'qw'},
    inplace=True
)
reconstructed_traj1 = pd.concat([common_traj1, gps_only_traj1, rts_only_traj1], ignore_index=True)
reconstructed_traj1.sort_values(by=['Timestamp'], inplace=True)
reconstructed_traj1.drop(['X_gps', 'Y_gps', 'Z_gps', 'qx_y', 'qy_y', 'qz_y', 'qw_y', 'PDOP', 'Sats', '_merge'], axis=1, inplace=True)

# %%
reconstructed_traj1.to_csv(BASE_PATH / "traj1/traj1_groundtruth_reconstructed.csv", index=False, header=False, sep = ' ')

# %%
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(common_traj1['X']-245610, common_traj1['Y']-5182375, s=1, c='lightgrey', label = 'common')
ax.scatter(gps_only_traj1['X']-245610, gps_only_traj1['Y']-5182375, s=3, c='lightskyblue', label = 'GPS only')
ax.scatter(rts_only_traj1['X']-245610, rts_only_traj1['Y']-5182375, s=1, c='lightsalmon', label = 'RTS only')
ax.set_aspect('equal')
ax.legend()
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_title('Trajectory reconstruction with RTS and GPS')
ax.set_aspect('equal')

fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(reconstructed_traj1['X']-245610, reconstructed_traj1['Y']-5182375, s=1, c='lightgrey', label = 'reconstructed')
ax.set_title('Reconstructed trajectory')
ax.set_xlabel('X [m]')
ax.set_ylabel('Y [m]')
ax.set_aspect('equal')
ax.legend()
plt.show()

# %%
df_imu = pd.read_csv(BASE_PATH / "trajectories/imu_odom_traj1.csv", names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
df_imu['Timestamp'] = pd.to_datetime(df_imu['Timestamp'], unit='s') # Convert to datetime

df_imu['Timestamp'] -= pd.Timedelta(hours=4, seconds=6) # Subtract 4 hours from the RTS Timestamp column
# print(df_imu.head())
df_imu['Timestamp'] = df_imu["Timestamp"].values.astype(np.int64) // (1*10 ** 9) # Convert to Unix Timestamp

df_imu.to_csv(BASE_PATH / "trajectories/imu_modified.csv", index=False, header=False, sep = ' ')

# %%











Notez quelque chose









