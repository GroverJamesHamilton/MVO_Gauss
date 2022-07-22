#include "ros/ros.h"
#include "sensor_msgs/Imu.h"

void chatterCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
  //ROS_INFO("Imu Seq: [%d]", msg->header.seq);
  ROS_INFO("Imu Orientation x: [%f], y: [%f], z: [%f], w: [%f]", msg->orientation.x,msg->orientation.y,msg->orientation.z,msg->orientation.w);
  ROS_INFO("Gyro x: [%f], y: [%f], z: [%f]", msg->angular_velocity.x,msg->angular_velocity.y,msg->angular_velocity.z);
  ROS_INFO("Imu acceleration x: [%f], y: [%f], z: [%f]", msg->linear_acceleration.x,msg->linear_acceleration.y,msg->linear_acceleration.z);
}

int main(int argc, char **argv)
{

  ros::init(argc, argv, "imu_node");
  ros::NodeHandle n;
  ros::Subscriber sub = n.subscribe("/kitti/oxts/imu", 1000, chatterCallback);
  ros::spin();

  return 0;
}
