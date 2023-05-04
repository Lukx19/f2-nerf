#ifndef NERF_BASED_LOCALIZER_HPP_
#define NERF_BASED_LOCALIZER_HPP_

#include <rclcpp/rclcpp.hpp>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>

#include <deque>

#include "localizer_core.hpp"

class NerfBasedLocalizer : public rclcpp::Node
{
public:
  NerfBasedLocalizer(
    const std::string & name_space = "",
    const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void callback_initial_pose(
    const geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr pose_conv_msg_ptr);
  void callback_image(const sensor_msgs::msg::Image::ConstSharedPtr image_msg_ptr);
  void publish_pose();

  // NerfBasedLocalizer subscribes to the following topics:
  // (1) initial_pose_with_covariance [geometry_msgs::msg::PoseWithCovarianceStamped]
  rclcpp::Subscription<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    initial_pose_with_covariance_subscriber_;
  // (2) image [sensor_msgs::msg::Image]
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscriber_;

  // NerfBasedLocalizer publishes the following topics:
  // (1) nerf_pose [geometry_msgs::msg::PoseStamped]
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr nerf_pose_publisher_;
  // (2) nerf_pose_with_covariance [geometry_msgs::msg::PoseWithCovarianceStamped]
  rclcpp::Publisher<geometry_msgs::msg::PoseWithCovarianceStamped>::SharedPtr
    nerf_pose_with_covariance_publisher_;

  std::string map_frame_;

  std::deque<geometry_msgs::msg::PoseWithCovarianceStamped::ConstSharedPtr>
    initial_pose_msg_ptr_array_;
  std::mutex initial_pose_array_mtx_;

  LocalizerCore localizer_core_;
};

#endif  // NERF_BASED_LOCALIZER_HPP_