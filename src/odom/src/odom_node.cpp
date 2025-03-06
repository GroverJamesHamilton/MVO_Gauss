#include <iostream>
#include "ros/ros.h"
#include "nav_msgs/Odometry.h"
#include <tf/transform_listener.h>

//using namespace cv;
using namespace std;

void chatterCallback(const nav_msgs::Odometry::ConstPtr& msg)
{
  //ROS_INFO("Seq: [%d]", msg->header.seq);
  //ROS_INFO("Position-> x: [%f], y: [%f], z: [%f]", msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
  //ROS_INFO(msg->pose.pose.position.x,msg->pose.pose.position.y, msg->pose.pose.position.z);
  cout << msg->pose.pose.position.x << ", " << msg->pose.pose.position.y << ";" << endl;
  //ROS_INFO("Orientation-> x: [%f], y: [%f], z: [%f], w: [%f]", msg->pose.pose.orientation.x, msg->pose.pose.orientation.y, msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
  //ROS_INFO("Vel-> Linear: [%f], Angular: [%f]", msg->twist.twist.linear.x,msg->twist.twist.angular.z);
}
int main(int argc, char **argv)
{
  cout << "testar robot_localization: " << endl;
  ros::init(argc, argv, "odom_node");
  ros::NodeHandle n;
  ros::NodeHandle node;
  //ros::Subscriber sub = n.subscribe("/myumi_005/base/state_controller/odom", 1000, chatterCallback);
  //ros::Subscriber sub = n.subscribe("/odometry/filtered", 1000, chatterCallback);
  ros::spin();
  return 0;
}
