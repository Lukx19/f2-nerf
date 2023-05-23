import glob
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose
from cv_bridge import CvBridge
import argparse
import time
from scipy.spatial.transform import Rotation
import pandas as pd
import tf2_ros
import geometry_msgs.msg
import tf2_geometry_msgs


class ImagePosePublisher(Node):

    def __init__(self, data_dir):
        super().__init__('image_pose_publisher')
        self.image_pub = self.create_publisher(Image, 'image', 10)
        self.pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, 'initial_pose_with_covariance', 10)
        self.bridge = CvBridge()

        self.subscription = self.create_subscription(
            PoseWithCovarianceStamped,
            '/nerf_pose_with_covariance',
            self.nerf_pose_with_covariance_callback,
            10)

        self.image_files = sorted(glob.glob(f"{data_dir}/images/*.png"))
        self.from_cams_meta = False
        self.pose_msg_list = list()
        if self.from_cams_meta:
            self.poses = np.load(f"{data_dir}/cams_meta.npy")
            for i in range(len(self.poses)):
                curr_pose = self.poses[i][0:12].reshape(3, 4)
                pose_msg = Pose()
                pose_msg.position.x = curr_pose[0, 3]
                pose_msg.position.y = curr_pose[1, 3]
                pose_msg.position.z = curr_pose[2, 3]
                rotation_mat = curr_pose[0:3, 0:3]
                r = Rotation.from_matrix(rotation_mat)
                q = r.as_quat()
                pose_msg.orientation.x = q[0]
                pose_msg.orientation.y = q[1]
                pose_msg.orientation.z = q[2]
                pose_msg.orientation.w = q[3]
                self.pose_msg_list.append(pose_msg)
        else:
            self.poses = pd.read_csv(
                f"{data_dir}/pose.tsv", sep="\t", index_col=0)
            self.image_files = self.image_files[0:len(self.poses)]
            for i in range(len(self.poses)):
                curr_pose = self.poses.iloc[i]
                pose_msg = Pose()
                pose_msg.position.x = curr_pose["x"]
                pose_msg.position.y = curr_pose["y"]
                pose_msg.position.z = curr_pose["z"]
                pose_msg.orientation.x = curr_pose["qx"]
                pose_msg.orientation.y = curr_pose["qy"]
                pose_msg.orientation.z = curr_pose["qz"]
                pose_msg.orientation.w = curr_pose["qw"]
                self.pose_msg_list.append(pose_msg)

        assert len(self.image_files) == len(self.poses), \
            f"Number of images ({len(self.image_files)}) and poses ({len(self.poses)}) do not match."

        self.offset = [0.705, 0.0, 0.262]

        # Publish tf
        self.tf_broadcaster = tf2_ros.StaticTransformBroadcaster(self)
        self.tf_msg = geometry_msgs.msg.TransformStamped()
        self.tf_msg.header.stamp = self.get_clock().now().to_msg()
        self.tf_msg.header.frame_id = "base_link"
        self.tf_msg.child_frame_id = "velodyne_front"
        self.tf_msg.transform.translation.x = self.offset[0]
        self.tf_msg.transform.translation.y = self.offset[1]
        self.tf_msg.transform.translation.z = self.offset[2]
        self.tf_msg.transform.rotation.x = 0.0
        self.tf_msg.transform.rotation.y = 0.0
        self.tf_msg.transform.rotation.z = 0.0
        self.tf_msg.transform.rotation.w = 1.0
        self.tf_broadcaster.sendTransform(self.tf_msg)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.shutdown_requested = False

        # Publish at 10 Hz
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.idx = 0

    def timer_callback(self):
        if self.idx == 0:
            self.publish_data()

    def publish_data(self):
        if self.idx >= len(self.image_files) or self.idx >= len(self.poses):
            self.get_logger().info('Finished publishing images and poses.')
            self.shutdown_requested = True
            return
        self.get_logger().info(f'Publishing images and poses {self.idx:04d}.')

        # Publish pose
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'map'
        pose_msg.pose.pose = self.pose_msg_list[self.idx]

        # Transform the pose_msg from the frame "velodyne_front" to the frame "base_link"
        try:
            transform = self.tf_buffer.lookup_transform(
                "velodyne_front", "base_link", rclpy.time.Time())
            pose_msg.pose.pose = tf2_geometry_msgs.do_transform_pose(
                pose_msg.pose.pose, transform)
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            self.get_logger().error('Failed to get transform from velodyne_front to base_link')
            return

        self.pose_pub.publish(pose_msg)

        # In the NeRF node, poses are held in a queue and processed when an image is subscribed.
        # If poses and images are published simultaneously, the pose might not be held in the queue and processed before the image is subscribed.
        # Therefore, a delay is introduced between publishing poses and images to ensure the proper processing order.
        time.sleep(0.025)

        # Publish image
        image = cv2.imread(self.image_files[self.idx])
        msg_image = self.bridge.cv2_to_imgmsg(image, 'bgr8')
        self.image_pub.publish(msg_image)

        self.idx += 1

    def nerf_pose_with_covariance_callback(self, msg):
        # Transform the pose_msg from the frame "velodyne_front" to the frame "base_link"
        # try:
        #     transform = self.tf_buffer.lookup_transform(
        #         "base_link", "velodyne_front", rclpy.time.Time())
        #     msg.pose.pose = tf2_geometry_msgs.do_transform_pose(msg.pose.pose, transform)
        # except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        #     self.get_logger().error('Failed to get transform from velodyne_front to base_link')
        #     exit(1)
        # pose = msg.pose.pose

        # if self.idx < len(self.pose_msg_list):
        #     self.pose_msg_list[self.idx] = pose
        self.publish_data()


def main(args=None):
    rclpy.init(args=args)
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, help='Path to data directory')
    args = parser.parse_args()

    node = ImagePosePublisher(args.data_dir)
    try:
        while rclpy.ok() and not node.shutdown_requested:
            rclpy.spin_once(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
