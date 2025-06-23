import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions, StorageFilter

import numpy as np
import argparse

def read_positions_from_ros2bag(bag_path, topic_name='/odom'):
    # Set up rosbag reader
    storage_options = StorageOptions(uri=bag_path, storage_id='sqlite3')
    converter_options = ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # Get message type for the topic
    topic_types = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topic_types}
    if topic_name not in type_map:
        raise ValueError(f"Topic {topic_name} not found in bag.")

    msg_type_str = type_map[topic_name]
    msg_type = get_message(msg_type_str)

    reader.set_filter(StorageFilter(topics=[topic_name]))

    positions = []
    timestamps = []

    while reader.has_next():
        topic, data, t = reader.read_next()
        msg = deserialize_message(data, msg_type)

        # Assuming nav_msgs/msg/Odometry
        pos = msg.pose.pose.position
        positions.append([pos.x, pos.y])
        timestamps.append(t * 1e-9)  # convert from nanoseconds to seconds

    return np.array(positions), np.array(timestamps)

def compute_average_distance(positions):
    diffs = np.diff(positions, axis=0)
    distances = np.linalg.norm(diffs, axis=1)
    return np.mean(distances)

def main():
    parser = argparse.ArgumentParser(description="Read positions from a ROS 2 bag and compute average distance between frames.")
    parser.add_argument('bag_path', help='Path to the ROS 2 bag directory (e.g., "my_bag")')
    parser.add_argument('--topic', default='/odom', help='Topic name to extract position data from (default: /odom)')
    args = parser.parse_args()

    rclpy.init()

    positions, timestamps = read_positions_from_ros2bag(args.bag_path, args.topic)

    if len(positions) < 2:
        print("Not enough data to compute distances.")
        rclpy.shutdown()
        return

    avg_dist = compute_average_distance(positions)
    print(f"Average distance between frames: {avg_dist:.4f} meters")

    rclpy.shutdown()

if __name__ == '__main__':
    main()
