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
from scipy.spatial.transform import Rotation as R


def split_rts_gps(df_gps, df_rts):

    df_gps['Timestamp'] = df_gps['Timestamp'].values.astype(np.float64)
    
    # Merge the two dataframes by timestamp
    merged_data = pd.merge_asof(
        df_rts[['Timestamp', 'T']],
        df_gps[['Timestamp', 'T']],
        on=['Timestamp'], tolerance=0.5, direction='nearest'
    ).dropna()

    # Find points that are only in the RTS trajectory
    merged_rts_common = pd.merge(
        merged_data[['Timestamp', 'T_x']],
        df_rts[['Timestamp', 'T']],
        on=['Timestamp'],
        how='outer',
        indicator=True
    )
    rts_only_traj = merged_rts_common[merged_rts_common['_merge'] == 'right_only']

    # Merge the two dataframes by timestamp in the other direction
    merged_data = pd.merge_asof(
        df_gps[['Timestamp', 'T']],
        df_rts[['Timestamp', 'T']],
        on=['Timestamp'], tolerance=0.2, direction='nearest'
    ).dropna()

    # Find points that are only in the GPS trajectory
    merged_gps_common = pd.merge(
        merged_data[['Timestamp', 'T_x']],
        df_gps[['Timestamp', 'T']],
        on=['Timestamp'], how='outer', indicator=True
    )
    gps_only_traj = merged_gps_common[merged_gps_common['_merge'] == 'right_only']

    
    # Clean the dataframes
    common_traj = merged_data.rename(columns={'T_x':'T_gps', 'T_y':'T_rts'})
    rts_only_traj.drop(['T_x', '_merge'], axis=1, inplace=True)
    gps_only_traj.drop(['T_x', '_merge'], axis=1, inplace=True)
    
    return gps_only_traj, rts_only_traj, common_traj


def generate_reconstructed_traj(gps_only_traj, rts_only_traj, common_traj):

    # Transform the common trajectory to the GPS frame
    P_common = np.array([
        [T[0,3] for T in common_traj['T_rts']], 
        [T[1,3] for T in common_traj['T_rts']],
        [T[2,3] for T in common_traj['T_rts']],
        np.ones(len(common_traj['T_rts']))
    ])
    Q_common = np.array([
        [T[0,3] for T in common_traj['T_gps']],
        [T[1,3] for T in common_traj['T_gps']],
        [T[2,3] for T in common_traj['T_gps']],
        np.ones(len(common_traj['T_gps']))
    ])
    T = ptp.minimization(P_common, Q_common)
    common_traj['T_rts'] = common_traj['T_rts'].apply(lambda x: T @ x)
    rts_only_traj['T'] = rts_only_traj['T'].apply(lambda x: T @ x)

    # Concat the common, gps only and rts only trajectories
    common_traj.rename(columns={'T_rts':'T'}, inplace=True)
    common_traj.drop(['T_gps'], axis=1, inplace=True)
    reconstructed_traj = pd.concat([common_traj, gps_only_traj, rts_only_traj], ignore_index=True)
    reconstructed_traj.sort_values(by=['Timestamp'], inplace=True)

    return reconstructed_traj, T


def pose_quat_to_tranform(dataframe):

    transforms = []
    for idx, pose in dataframe.iterrows():
        r = R.from_quat([pose['qx'], pose['qy'], pose['qz'], pose['qw']])
        R_mat = r.as_matrix()
        T = np.eye(4)
        T[:3,:3] = R_mat
        T[:3,3] = np.array([pose['X'], pose['Y'], pose['Z']])
        transforms.append(T)

    dataframe['T'] = transforms
    dataframe.drop(['X', 'Y', 'Z', 'qx', 'qy', 'qz', 'qw'], axis=1, inplace=True)
    return dataframe


def transform_to_pose_quat(dataframe):

    dataframe['X'] = [T[0,3] for T in dataframe['T']]
    dataframe['Y'] = [T[1,3] for T in dataframe['T']]
    dataframe['Z'] = [T[2,3] for T in dataframe['T']]
    rot = [R.from_matrix(T[:3,:3]) for T in dataframe['T']]
    dataframe['qx'] = [r.as_quat()[0] for r in rot]
    dataframe['qy'] = [r.as_quat()[1] for r in rot]
    dataframe['qz'] = [r.as_quat()[2] for r in rot]
    dataframe['qw'] = [r.as_quat()[3] for r in rot]
    dataframe.drop(['T'], axis=1, inplace=True)
    return dataframe


def plot_reconstructed_traj(df_gps, df_rts, gps_only_traj, rts_only_traj, common_traj, arrows=False):

    X_origin = 245610
    Y_origin = 5182375

    plt.rc('font', family='serif', serif='Times')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=20)
    plt.rc('ytick', labelsize=20)
    plt.rc('axes', labelsize=20)
    plt.rcParams.update({'font.size': 20})
    font_size = 20

    
    fig ,axs = plt.subplots(1,3,figsize =(15,5))
    fig.subplots_adjust(left=.13, bottom=.13, right=.99, top=.99,wspace=3)
    colors = ['lightskyblue', 'lightsalmon', 'lightgrey']
    labels = ['GPS', 'RTS', 'Commune']
    trajs = [gps_only_traj, rts_only_traj, common_traj]
    titles = ['GNSS', 'RTS', 'Trajectoire reconstruite']

    axs[0].scatter([T[0,3] for T in df_gps['T'] - X_origin], [T[1,3] for T in df_gps['T'] - Y_origin], s=2, c=colors[0], label = labels[0])
    axs[1].scatter([T[0,3] for T in df_rts['T'] - X_origin], [T[1,3] for T in df_rts['T'] - Y_origin], s=2, c=colors[1], label = labels[1])
    for i,traj in enumerate(trajs) :
        X = [T[0,3] for T in traj['T'] - X_origin]
        Y = [T[1,3] for T in traj['T'] - Y_origin]
        axs[2].scatter(X, Y, s=2, c=colors[i], label = labels[i])

        if arrows:
            nx_u = [T[0,0] for T in traj['T']]
            nx_v = [T[1,0] for T in traj['T']]
            ny_u = [T[0,1] for T in traj['T']]
            ny_v = [T[1,1] for T in traj['T']]
            axs[2].quiver(X, Y, nx_u, nx_v, color='r', scale=1, scale_units='xy', angles='xy', headwidth=1)
            axs[2].quiver(X, Y, ny_u, ny_v, color='g', scale=1, scale_units='xy', angles='xy', headwidth=1)

    last_ax = axs.flat[-1]
    x_lim = last_ax.get_xlim()
    y_lim = last_ax.get_ylim()

    for ax, title in zip(axs.flat, titles):
        ax.title.set_text(title)
        ax.set_aspect('equal')
        # ax.legend(loc='lower left')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        for axis in ['top','bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(1.5)

    fig.set_size_inches(13, 5.5)
    
def find_transform_between_origins(gt_traj, df_icp):

    merged_data = pd.merge_asof(
        gt_traj[['Timestamp', 'T']],
        df_icp[['Timestamp', 'T']],
        on=['Timestamp'], tolerance=0.5, direction='nearest'
    ).dropna()

    #T icp vers gps
    unit_vectors = np.eye(3)
    unit_vectors = np.vstack((unit_vectors, np.array([1,1,1])))

    P_gt = merged_data['T_x'].iloc[0] @ unit_vectors
    P_icp = merged_data['T_y'].iloc[0] @ unit_vectors
    T = ptp.minimization(P_icp, P_gt)

    return T


def plot_icp_vs_gt(icp_traj, gt_traj, arrows=False):

    X_origin = 245610
    Y_origin = 5182375

    fig, ax = plt.subplots(1,1,figsize =(6,6))
    colors = ['lightskyblue', 'mediumseagreen']
    labels = ['ICP', 'Vérité terrain']
    trajs = [icp_traj, gt_traj]
    for traj, color, label in zip(trajs, colors, labels):
        X = [T[0,3] for T in traj['T'] - X_origin]
        Y = [T[1,3] for T in traj['T'] - Y_origin]
        if arrows:
            nx_u = [T[0,0] for T in traj['T']]
            nx_v = [T[1,0] for T in traj['T']]
            ny_u = [T[0,1] for T in traj['T']]
            ny_v = [T[1,1] for T in traj['T']]
            ax.quiver(X, Y, nx_u, nx_v, color='r', scale=1, scale_units='xy', angles='xy', headwidth=1)
            ax.quiver(X, Y, ny_u, ny_v, color='g', scale=1, scale_units='xy', angles='xy', headwidth=1)

        ax.scatter(X, Y, s=2, c=color, label = label)
        # ax.title.set_text('ICP trajectory comparaison with groundtruth')
        ax.title.set_text('Comparaison de la trajectoire ICP avec la vérité terrain')
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_aspect('equal')
        ax.legend(loc='lower left')


def main(gps_file, rts_file, icp_file, plot, save_plot, arrows, save, verbose):

    df_gps = pd.read_csv(BASE_PATH / f'{gps_file}', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
    df_rts = pd.read_csv(BASE_PATH / f'{rts_file}', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')
    df_icp = pd.read_csv(BASE_PATH / f'{icp_file}', names=["Timestamp", "X", "Y", "Z", "qx", "qy", "qz", "qw"], delimiter= ' ')

    for df in [df_gps, df_rts, df_icp]:
        df = pose_quat_to_tranform(df)
    gps_only_traj, rts_only_traj, common_traj = split_rts_gps(df_gps, df_rts)
    reconstructed_traj, T = generate_reconstructed_traj(gps_only_traj, rts_only_traj, common_traj)
    df_rts_transformed = df_rts.copy()
    df_rts_transformed['T'] = df_rts_transformed['T'].apply(lambda x: T @ x)

    if plot:
        plot_reconstructed_traj(df_gps, df_rts_transformed, gps_only_traj, rts_only_traj, common_traj, arrows=arrows)
        plt.tight_layout()
        if save_plot :
            plt.savefig(BASE_PATH / 'figures' / 'reconstructed_traj.pdf')
        plt.show()

    T_icp_gt = find_transform_between_origins(reconstructed_traj, df_icp)
    df_icp['T'] = df_icp['T'].apply(lambda x: T_icp_gt @ x)

    if plot:
        plot_icp_vs_gt(df_icp, reconstructed_traj, arrows=arrows)
        plt.tight_layout()
        if save_plot :
            plt.savefig(BASE_PATH / 'figures' / 'comparison_icp_gt.pdf')
        plt.show()

    if save:
        df_icp = transform_to_pose_quat(df_icp)
        reconstructed_traj = transform_to_pose_quat(reconstructed_traj)
        df_icp.to_csv(BASE_PATH / 'output' / 'icp_reconstructed_traj.csv', index=False, header=False, sep = ' ')
        reconstructed_traj.to_csv(BASE_PATH / 'output' / 'gt_reconstructed_traj.csv', index=False, header=False, sep = ' ')
    
    if verbose :
        print('Nombre de poses de la trajectoire uniquement GNSS :',len(gps_only_traj))
        print('Nombre de poses de la trajectoire uniquement RTS :',len(rts_only_traj))
        print('Nombre de poses de la trajectoire uniquement commune :',len(common_traj))
        print('Nombre de poses de la trajectoire reconstruite :',len(reconstructed_traj))




def init_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument('-gps',
                        type=str, required=True,
                        help='Name of gps file.')
    parser.add_argument('-rts',
                        type=str, required=True,
                        help='Name of rts file.')
    parser.add_argument('-icp',
                        type=str, required=True,
                        help='Name of icp file.')
    parser.add_argument('-p', '--plot', action='store_true',
                        help='Plot trajectories for debugging.')
    parser.add_argument('--save_plot', action='store_true',
                        help='save plot in results folder.')
    parser.add_argument('-a', '--arrows', action='store_true',
                        help='Plot arrows onto trajectories.')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save resulting trajectories in output folder.')
    parser.add_argument('--verbose', action='store_true',
                        help='Print informations about reconstructed trajectory.')
    return parser


if __name__ == '__main__':

    parser = init_argparse()
    args = parser.parse_args()
    main(args.gps, args.rts, args.icp, args.plot, args.save_plot, args.arrows, args.save, args.verbose)