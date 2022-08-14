#include <iostream>
#include <vector>

//ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h>

//Odometry
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

//CV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>

//Own-made functions from function.cpp
#include "function.h"

double it = 1;
Mat oldFrame, oldFrame0, oldFrame1, oldFrame2, frame, frame0, frame1, frame2, crop, oldCrop;

//Below are the rotation and translation matrices used for different purposes
Mat R = cv::Mat::eye(3,3,CV_64F);
Mat R0 = cv::Mat::eye(3,3,CV_64F);
double tran[3][1] = {{0},{0},{0}};
Mat Rpos = cv::Mat::eye(3,3,CV_64F);
Mat Rprev = cv::Mat::eye(3,3,CV_64F);
Mat tpos = Mat(3,1,CV_64F,tran);
Mat ttemp = tpos;
Mat rot = cv::Mat::eye(3,3,CV_64F);
Mat rotDiff = cv::Mat::eye(3,3,CV_64F);
Mat t = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat tprev = cv::Mat::zeros(cv::Size(1,3), CV_64F);

Mat PkHat;
Mat Pk_1Hat = cv::Mat::eye(cv::Size(4,3), CV_64F);

vector<KeyPoint> keyp1, keyp2;
Mat desc1, desc2;
vector<DMatch> matches;
cv::Mat point3d;
vector<Point2d> scene1, scene2;

float yaw = 0;
float yawDiff = 0;
float yawVel;

int avgMatches = 0;
int nrMatches = 0;

double dim = 700;
double dimShow = 700;
double showScale = dimShow/dim;
double curScale = 1.4;
double prevScale = 1.4;
double alpha = 0.1;
double camHeight = 1.65;
double xOffset = 414;
double yOffset = 175;
cv::Rect crop_region(xOffset, yOffset, 414, 200); //Only a small subset of the frame is extracted

//The printed trajectory
Mat trajectory = Mat::zeros(dim, dim, CV_8UC3);
Mat traj;
int X,Y; //Pixel location to print trajectory
double x,y,vx,vy;

Ptr<ORB> orbis = cv::ORB::create(1500,
		1.25f,
		8,
		16,
		0,
		3,
		ORB::FAST_SCORE,
	  31, // Descriptor patch size
		9);
//The camera matrix dist. coeffs for the KITTI dataset
	double kittiCamMatrix[3][3] = {
	  {959.791,0,696.0217 - xOffset}, //
	  {0,956.9251,224.1806 - yOffset},
	  {0,0,1}};

	Mat K = Mat(3,3,CV_64F,kittiCamMatrix);
  double distCoeffs[8][1] = {-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705};
  Mat dist = Mat(8,1,CV_64F,distCoeffs);

	ros::Time delay(0.02);

// Ros time stamp
ros::Time simTime, lastTime;
using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image Window";

void infoCb(const sensor_msgs::CameraInfoConstPtr& info_msg)
{
	//cout << info_color_msg->header.stamp << endl;
	simTime = info_msg->header.stamp;
	//cout << "Last time: " << lastTime << endl;
	cout << "Sim real time: " << simTime << endl;
}
class ImageConverter
{
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/kitti/camera_color_left/image_raw", 1, &ImageConverter::imageCb, this);
    cv::namedWindow(OPENCV_WINDOW);
  }
  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
		cout << "Sim time: " << ros::Time::now() << endl;
    cv_bridge::CvImagePtr cv_ptr;
    try
    {
      cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
    }
    catch (cv_bridge::Exception& e)
    {
      ROS_ERROR("cv_bridge exception: %s", e.what());
      return;
    }
	//-------------------------------------------------------------------------------------------
  //You can test your algorithm here.
	R.copyTo(Rprev);
	t.copyTo(tprev);
  if (it == 1)
  {
    oldFrame0 = cv_ptr->image;
    undistort(oldFrame0, oldFrame, K, dist, Mat());
		//undistort(oldFrame0, oldFrame, Kitti, Mat(), Mat());
		oldCrop = oldFrame(crop_region);
    orbis->detectAndCompute(oldCrop, noArray(), keyp1, desc1, false);
  }
    frame0 = cv_ptr->image;
    undistort(frame0, frame, K, dist, Mat());
		//undistort(frame0, frame, Kitti, Mat(), Mat());
		crop = frame(crop_region);
    orbis->detectAndCompute(crop, noArray(), keyp2, desc2, false);

    if(keyp1.size() > 6 || keyp2.size() > 6)
    {
    matches = BruteForce(oldCrop, crop, keyp1, keyp2, desc1, desc2, 0.5);
    tie(t,R) = tranRot(keyp1, keyp2, matches);

		nrMatches = nrMatches + matches.size();
		avgMatches = nrMatches/it;
		double avgDist = avgMatchDist(matches);
		//cout << endl;

		PkHat = scaleUpdate(K, Rprev, R, tprev, t);
		tie(scene1, scene2) = getPixLoc(keyp1, keyp2, matches);
		cv::triangulatePoints(K*Pk_1Hat, PkHat, scene1, scene2, point3d);

		if(matches.size() < 500)
		{
		curScale = getScale(point3d, PkHat, matches, keyp2, prevScale, alpha, camHeight);
		prevScale = curScale;
	  }
		auto velocity = curScale/0.1; //Scale is the total displacement in meters, over the time of 100ms the velocity can be found
																	//Solution is not optimal but works if calculation time
		//cout << "Total velocity: " << velocity << " m/s" << endl; //Display total velocity
		//cout << velocity << endl;
		if(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1 && avgDist > 10)//Safeguard to in case if image is still
		{
		Rodrigues(Rpos, rot, noArray()); //Converts rotation matrix to rotation vector
		Rodrigues(R, rotDiff, noArray());//Same as above but with the difference in Yaw
		yawDiff = rotDiff.at<double>(1,0);
		yaw = rot.at<double>(1,0);
		tpos = tpos + Rpos*t*curScale; //The scaled estimate is updated here
		Rpos = R*Rpos;								 //The rotation matrix updated after

		}
		if(it > 1)
		{
    X = 2*tpos.at<double>(0,0);
    Y = 2*tpos.at<double>(0,2);

		x = -tpos.at<double>(0,2);
		y = tpos.at<double>(0,0);

		vx = -t.at<double>(0,2)*velocity;
		vy = t.at<double>(0,0)*velocity;

		yawVel = yawDiff/0.1;

		//cout << "Xpos: " << x << endl;
		//cout << "Ypos: " << y << endl;
		//cout << "xvec: " << vx << endl;
		//cout << "yvec: " << vy << endl;
		//cout << "Yaw: " << yaw << endl;
		//cout << "Yaw difference: " << yawDiff*180/3.14159 << endl;
		//cout << x << ", " << y << ";" << endl;
	  }

		//ros::Subscriber info_sub_ = nInfo.subscribe("/kitti/camera_color_left/camera_info", 1, infoCb);
		//Quaternion created from yaw to publish in nav_msgs

    geometry_msgs::Quaternion odom_quat = tf::createQuaternionMsgFromYaw(yaw);

		//first, we'll publish the transform over tf
		geometry_msgs::TransformStamped odom_trans;
		//odom_trans.header.stamp = current_time;
		odom_trans.header.frame_id = "odom";
		odom_trans.child_frame_id = "base_link";

		odom_trans.transform.translation.x = x;
		odom_trans.transform.translation.y = y;
		odom_trans.transform.translation.z = 0.0;
		odom_trans.transform.rotation = odom_quat;

		//We'll publish the odometry message over ROS
    odom.header.frame_id = "odom";
		odom.header.stamp = simTime;
    //set the position
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = odom_quat;
		odom.twist.twist.linear.x = vx;
		odom.twist.twist.linear.y = vy;
		odom.twist.twist.angular.z = yawVel;

		//cout << "Yaw: " << yaw << endl;
		//cout << "Average number of matches: " << avgMatches << endl;
		//cout << "Average match distance: " << avgDist << endl;

    //publish the message
    odom_pub_.publish(odom);

    //cout << "Iterations: " << it << endl;
		circle(trajectory, Point(X + dim/2,Y + dim/2), 2, Scalar(0,0,255), 2);
    cv::resize(trajectory, traj, cv::Size(), showScale, showScale);
    imshow("Trajectory", traj);
    }
		oldCrop = crop.clone();
  	oldFrame = frame.clone();
    keyp1 = keyp2;
    desc1 = desc2.clone();
		//cout << "Frame number: " << it << endl;
    it++;
		if(simTime == lastTime)
		{
			//cout << "Out of syncxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" << endl;
		}
		//cout << "Odom sim time: " << simTime << endl;
		cout << endl;
    //------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------
    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, frame);
    cv::waitKey(1);
    // Output modified video stream
    //image_pub_.publish(cv_ptr->toImageMsg());

  }
private:
  ros::NodeHandle nh_;
	ros::NodeHandle n;
	ros::NodeHandle nInfo;
	image_transport::Subscriber info_sub_;
	ros::Publisher odom_pub_ = n.advertise<nav_msgs::Odometry>("odom", 1);
	nav_msgs::Odometry odom;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
	ros::NodeHandle nInfo;
  ImageConverter ic;
  ros::spin();
  return 0;
}
