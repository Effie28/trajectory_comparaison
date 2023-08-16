#! /usr/bin/python3

import sys
import os
import argparse
from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from tqdm import tqdm
import pandas as pd
from pathlib import Path


################### PARAMETERS #####################

BASE_PATH = Path(__file__).absolute().parents[1]

####################################################


def main(bag_name, output_name, topic_name):

    input_filepath = BASE_PATH / "data" / bag_name / f"{bag_name}.db3"
    save_path = Path(BASE_PATH / "trajectories")

    if not os.path.isdir(input_filepath):
        print('Error: Cannot locate input bag file [%s]' % input_filepath, file=sys.stderr)
        sys.exit(2)

    bag_file = Reader(input_filepath)
    if topic_name not in bag_file.topics():
        print(f"Error: Cannot find topic {topic_name} in bag file {input_filepath}", file=sys.stderr)
    df = pd.DataFrame({"timestamp": [], "x": [], "y": [], "z": [], "q_x": [], "q_y": [], "q_z": [], "q_w": []})
    for conn, timestamp, data in tqdm(bag_file.messages()):
        if conn.topic == topic_name:
            try:
                msg = deserialize_cdr(data, conn.msgtype)
                df["timestamp"].append(msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9)
                df["x"].append(msg.pose.pose.position.x)
                df["y"].append(msg.pose.pose.position.y)
                df["z"].append(msg.pose.pose.position.z)
                df["q_x"].append(msg.pose.pose.orientation.x)
                df["q_y"].append(msg.pose.pose.orientation.y)
                df["q_z"].append(msg.pose.pose.orientation.z)
                df["q_w"].append(msg.pose.pose.orientation.w)
            except:
                print("Error: Unable to deserialize messages from desired topic.")
                sys.exit(3)
        
    if not os.path.isdir(str(save_path)): os.makedirs(str(save_path))
    output_filepath = save_path / f"{output_name}.csv"
    df.to_csv(output_filepath, index=False, sep=" ", header=None)
    print(f"Done. Saved trajectory to {output_filepath}")


def init_argparse():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input",
                        type=str, required=True,
                        help="Name of input rosbag.")
    parser.add_argument("-t", "--topic",
                        type=str, required=True,
                        help="Topic to save (need to be Odometry msgs).")
    parser.add_argument("-o", "--output",
                        type=str, required=True,
                        help="Output name for csv file in TUM format. Will be saved in 'trajectories' folder.")
    return parser


if __name__ == '__main__':
    parser = init_argparse()
    args = parser.parse_args()
    main(args.input, args.output, args.topic)