#include "ros/ros.h"
#include "sensor_msgs/Imu.h"
#include "sensor_msgs/NavSatFix.h"
#include <iostream>

void imuCallback(const sensor_msgs::Imu::ConstPtr& msg)
{
  //ROS_INFO("Imu Seq: [%d]", msg->header.seq);
  ROS_INFO("Imu Orientation x: [%f], y: [%f], z: [%f], w: [%f]", msg->orientation.x,msg->orientation.y,msg->orientation.z,msg->orientation.w);
}
void navCallback(const sensor_msgs::NavSatFix::ConstPtr& msg)
{
  ROS_INFO("Gps: lat: [%f], long: [%f]", msg->latitude,msg->longitude);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "fusion");
  ros::NodeHandle n;
  ros::Subscriber imu_sub = n.subscribe("/kitti/oxts/imu", 1000, imuCallback);

  ros::init(argc, argv, "nav_node");
  ros::NodeHandle m;
  ros::Subscriber nav_sub = m.subscribe("/kitti/oxts/gps/fix", 1000, navCallback);

  ros::spin();

  return 0;
}
