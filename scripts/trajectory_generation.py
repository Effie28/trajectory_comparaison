import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
BASE_PATH = Path(__file__).absolute().parents[1] 
import argparse
import point_to_point as ptp

def main(input_file, reference_file) :
    df_gps = pd.read_csv(BASE_PATH / f'{input_file}.csv', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
    df_rts = pd.read_csv(BASE_PATH / f'{reference_file}.csv', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')

    df_gps['Timestamp'] = df_gps['Timestamp'].values.astype(np.float64)

    merged_data = pd.merge_asof(
        df_gps[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
        df_rts[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
        on=['Timestamp'],
        tolerance=0.2,
        direction='nearest'
    ).dropna()

    merged_gps_common = pd.merge(
        merged_data[['Timestamp', 'X_x', 'Y_x', 'Z_x', 'qx_x','qy_x','qz_x','qw_x']],
        df_gps[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
        on=['Timestamp'],
        how='outer',
        indicator=True)
    
    common_traj = merged_data.rename(
        columns={'X_x':'X_gps', 'Y_x':'Y_gps', 'Z_x':'Z_gps', 'qx_x' : 'qx_gps', 'qy_x' : 'qy_gps', 'qz_x' : 'qz_gps', 'qw_x' : 'qw_gps',
        'X_y':'X_rts', 'Y_y':'Y_rts', 'Z_y':'Z_rts', 'qx_y' : 'qx_rts', 'qy_y' : 'qy_rts', 'qz_y' : 'qz_rts', 'qw_y' : 'qw_rts'},
        )
    gps_only_traj = merged_gps_common[merged_gps_common['_merge'] == 'right_only']
    gps_only_traj.drop(['X_x', 'Y_x', 'Z_x', 'qx_x', 'qy_x', 'qz_x', 'qw_x', '_merge'], axis=1, inplace=True)

    merged_data = pd.merge_asof(
        df_rts[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
        df_gps[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
        on=['Timestamp'],
        tolerance=0.5,
        direction='nearest'
    ).dropna()

    merged_rts_common = pd.merge(
        merged_data[['Timestamp', 'X_x', 'Y_x', 'Z_x', 'qx_x','qy_x','qz_x','qw_x']],
        df_rts[['Timestamp', 'X', 'Y', 'Z', 'qx','qy','qz','qw']],
        on=['Timestamp'],
        how='outer',
        indicator=True
    )

    rts_only_traj = merged_rts_common[merged_rts_common['_merge'] == 'right_only']
    rts_only_traj.drop(['X_x', 'Y_x', 'Z_x', 'qx_x', 'qy_x', 'qz_x', 'qw_x', '_merge'], axis=1, inplace=True)

    fig ,axs = plt.subplots(1,3,figsize =(24,8))
    axs[0].scatter(rts_only_traj['X'], rts_only_traj['Y'], s=1, c='lightgrey', label = 'RTS')
    axs[0].title.set_text('RTS only trajectory')
    axs[1].scatter(gps_only_traj['X'], gps_only_traj['Y'], s=3, c='lightskyblue', label = 'GPS')
    axs[1].title.set_text('GPS only trajectory')
    axs[2].scatter(common_traj['X_rts'], common_traj['Y_rts'], s=3, c='red', label = 'common')
    axs[2].title.set_text('Common trajectory')
    plt.show()

    #Transform the common trajectory to the GPS frame
    P_common = np.array([common_traj['X_rts'], common_traj['Y_rts'], common_traj['Z_rts'], np.ones(len(common_traj['X_rts']))])
    Q_common = np.array([common_traj['X_gps'], common_traj['Y_gps'], common_traj['Z_gps'], np.ones(len(common_traj['X_gps']))])
    T = ptp.minimization(P_common, Q_common)
    P_common_transformed = T @ P_common
    common_traj['X_rts'] = P_common_transformed[0,:]
    common_traj['Y_rts'] = P_common_transformed[1,:]
    common_traj['Z_rts'] = P_common_transformed[2,:]

    #Transform the RTS only trajectory to the GPS frame
    P_rts_only_traj = np.array([rts_only_traj['X'], rts_only_traj['Y'], rts_only_traj['Z'], np.ones(len(rts_only_traj['X']))])
    P_rts_only_traj1_transformed = T @ P_rts_only_traj
    rts_only_traj['X'] = P_rts_only_traj1_transformed[0,:]
    rts_only_traj['Y'] = P_rts_only_traj1_transformed[1,:]
    rts_only_traj['Z'] = P_rts_only_traj1_transformed[2,:]

    #Concat the common, gps only and rts only trajectories
    common_traj.rename(
        columns={'X_rts':'X', 'Y_rts':'Y', 'Z_rts':'Z', 'qx_rts' : 'qx', 'qy_rts' : 'qy', 'qz_rts' : 'qz', 'qw_rts' : 'qw'},
        inplace=True
    )
    reconstructed_traj = pd.concat(
        [common_traj, gps_only_traj, rts_only_traj],
          ignore_index=True
    )
    reconstructed_traj.sort_values(by=['Timestamp'], inplace=True)
    reconstructed_traj.drop(['X_gps', 'Y_gps', 'Z_gps', 'qx_gps', 'qy_gps', 'qz_gps', 'qw_gps'], axis=1, inplace=True)

    reconstructed_traj.to_csv(BASE_PATH / 'trajectories' / 'gt_reconstructed_traj1.csv', index=False, header=False, sep = ' ')

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(common_traj['X'], common_traj['Y'], s=1, c='lightgrey', label = 'common')
    ax.scatter(gps_only_traj['X'], gps_only_traj['Y'], s=3, c='lightskyblue', label = 'GPS only')
    ax.scatter(rts_only_traj['X'], rts_only_traj['Y'], s=1, c='lightsalmon', label = 'RTS only')
    ax.set_aspect('equal')
    ax.legend()
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_title('Trajectory reconstruction with RTS and GPS')
    ax.set_aspect('equal')

    plt.show()

def init_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        type=str, required=True,
                        help='Name of gps file.')
    parser.add_argument('-r', '--reference',
                        type=str, required=True,
                        help='Name of rts file.')
    return parser

if __name__ == '__main__':

    parser = init_argparse()
    args = parser.parse_args()
    main(args.input, args.reference)