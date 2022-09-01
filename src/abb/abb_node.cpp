#include <iostream>
#include <vector>

// ROS
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/CameraInfo.h> //

// Odometry node handling
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/Twist.h>

// OpenCV
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core.hpp>
#include <opencv2/calib3d.hpp>
//Own-made functions from function.cpp
#include "function.h"
//#include <typeinfo> //

//For tf, ground truth data in KITTI datasets are stored here
#include <tf/transform_listener.h>
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include <tf2_ros/transform_listener.h>

double it = 1;
Mat oldIm, oldIm0, Im, Im0, crop, oldCrop;
Mat scaleIm, oldScaleIm;
//Below are the rotation and translation matrices used for different purposes
Mat R = cv::Mat::eye(3,3,CV_64F); // Rotation matrix from epipolar geometry between 2 frames
Mat Rpos = cv::Mat::eye(3,3,CV_64F); // Rotation matrix
Mat Rprev = cv::Mat::eye(3,3,CV_64F);
Mat tpos = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat rot = cv::Mat::eye(3,3,CV_64F);
Mat rotDiff = cv::Mat::eye(3,3,CV_64F);
Mat t = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat tprev = cv::Mat::zeros(cv::Size(1,3), CV_64F);

cv::Mat PkHat, Pk;
Mat Pk_1Hat = cv::Mat::eye(cv::Size(4,3), CV_64F);
vector<KeyPoint> keyp1, keyp2, keyp3, keyp4;
Mat desc1, desc2, desc3, desc4;
vector<DMatch> matchesOdom, matches2;
cv::Mat point3d, pointX;
vector<Point2d> scene1, scene2, scene3, scene4;
//For publishing the yaw parameters
float yaw, yawDiff, yawVel;

double sampleTime = 0.103; //The sample time for the rosbag dataset in ms, default would be 100 or 100/3
//Display parameters
double dim = 350;
double dimShow = 700;
double showScale = dimShow/dim;
//Scale recovery parameters
double curScale = 0.9;
double prevScale = 0.9;
double alpha = 1; //Scale smoothing parameters, alpha = [0 1], 1 is no smoothing
double camHeight = 1.65; //Camera height, for scale recovery
//Frame size and cropping parameters:
//Odometry frame
int xpix = 1240;
int ypix = 375;
int xO1 = 0;
int yO1 = 0;
cv::Rect cropOdom(xO1, yO1, xpix - 2*xO1, ypix - yO1);
//Scale recovery frame
int xO2 = 200;
int yO2 = 200;
//cv::Rect cropScale(xO2, yO2, xpix - 2*xO2, ypix - yO2); //Only a small subset of the frame is extracted
cv::Rect cropScale(xO2, yO2, 500, ypix - yO2); //Frame for least scale recovery error, depends on the
//The printed trajectory, Mainly for visually
Mat trajectory = Mat::zeros(dim, dim, CV_8UC3);
Mat traj;
int X,Y; //Pixel location to print trajectory
double x,y,vx,vy;
//To determine if the camera is still between images, depends on dataset
double avgMatchesDist;
double avgDistThresh = 10;

//Initialize the ORB detectors/descriptors
//For the unscaled visual odometry
//Primarily: The max numbers of features and FAST threshold are the most crucial parameters
//to change since they have the most impact

Ptr<ORB> orbOdom = cv::ORB::create(1000, //Max number of features
		1.25f,															 //Pyramid decimation ratio > 1
		8,																	 //Number of pyramid levels
		16,																	 //Edge threshold
		0,																	 //The level of pyramid to put source image to
		3,																	 //WTA_K
		ORB::FAST_SCORE,										 //Which algorithm is used to rank features, either FAST_SCORE or HARRIS_SCORE
	  31,                                  //Descriptor patch size
		6);                                  //The FAST threshold
//For the scale recovery
Ptr<ORB> orbScale = cv::ORB::create(500,
		1.2f,
		8,
		16,
		0,
		3,
		ORB::HARRIS_SCORE,
		31, // Descriptor patch size
		10);
//The camera matrix dist. coeffs for the KITTI dataset
	double kOdom[3][3] = {
	  {959.791,0,696.0217 - xO1}, //Principal point needs to be offset if image is cropped
	  {0,956.9251,224.1806 - yO1},
	  {0,0,1}};

 double kScale[3][3] = {
		{959.791,0,696.0217 - xO2}, //Principal point needs to be offset if image is cropped
		{0,956.9251,224.1806 - yO2},
		{0,0,1}};

	Mat K = Mat(3,3,CV_64F,kOdom);
	Mat Kscale = Mat(3,3,CV_64F,kScale);
  double distCoeffs[5][1] = {-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705};
  Mat dist = Mat(5,1,CV_64F,distCoeffs);

// Ros time stamp
ros::Time simTime, lastTime;
double timeDiff;
using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image Window";

double xGT, yGT, zGT, xGT2, yGT2, scaleGT; //Ground truth parameters from node /tf
double yawGT, yawDiffGT;
double yawprevGT = -3.00251;

double velocity;

bool scaleRecoveryMode = false;

//Ground truth callback: Fetches values from node /tf
void tfCb(const tf2_msgs::TFMessage::ConstPtr& tf_msg)
{
//Ground truth translation
xGT = tf_msg->transforms.at(0).transform.translation.x;
yGT = tf_msg->transforms.at(0).transform.translation.y;
zGT = tf_msg->transforms.at(0).transform.translation.z;
//Ground truth rotation
auto rotw = tf_msg->transforms.at(0).transform.rotation.w;
auto rotx = tf_msg->transforms.at(0).transform.rotation.x;
auto roty = tf_msg->transforms.at(0).transform.rotation.y;
auto rotz = tf_msg->transforms.at(0).transform.rotation.z;
auto q = tf_msg->transforms.at(0).transform.rotation;
double yawGT = tf::getYaw(q);
auto yawDiffGT = yawGT - yawprevGT;
//Ground truth scale
scaleGT = sqrt(pow((xGT - xGT2),2)+pow((yGT - yGT2),2));
//cout << "Scale GT: " << scaleGT << endl;

//Prints ground truth to trajectory
circle(trajectory, Point(yGT + dim/2,xGT + dim/2), 1, Scalar(124,252,0), 1);
//Saves previous parameters of interest
yawprevGT = yawGT;
xGT2 = xGT;
yGT2 = yGT;
}

class ImageConverter
{
public:
  ImageConverter()
    : it_(nh_)
  {
    // Subscribe to input video feed and publish output video feed
		image_sub_ = it_.subscribe("/kitti/camera_color_left/image_raw", 50, &ImageConverter::imageCb, this);
		//image_sub_ = it_.subscribe("/kitti/camera_gray/right/image_rect", 50, &ImageConverter::imageCb, this);
		//image_sub_ = it_.subscribe("/kitti/camera_gray/right/image_rect", 1, &ImageConverter::imageCb, this);
    cv::namedWindow(OPENCV_WINDOW);
  }
  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }
	//Image callback, where the main algorithm is
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
		simTime = ros::Time::now();
		timeDiff = (simTime - lastTime).toSec(); //Simulation time difference, i.e. how long an iteration of the algorithm takes
		//cout << "Time difference: " << timeDiff << endl;
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
  //Algorithm starts here

  if (it == 1) //First frame to initialize
  {
    oldIm0 = cv_ptr->image;
    undistort(oldIm0, oldIm, K, dist, Mat());
		oldCrop = oldIm(cropOdom); //
    orbOdom->detectAndCompute(oldCrop, noArray(), keyp1, desc1, false);

		oldScaleIm = oldIm(cropScale);
		if(scaleRecoveryMode)
		{
		orbScale->detectAndCompute(oldScaleIm, noArray(), keyp3, desc3, false);
	  }
  }
    Im0 = cv_ptr->image;
    undistort(Im0, Im, K, dist, Mat());
		crop = Im(cropOdom);
    orbOdom->detectAndCompute(crop, noArray(), keyp2, desc2, false);

		scaleIm = oldIm(cropScale);
		if(scaleRecoveryMode)
		{
		orbScale->detectAndCompute(scaleIm, noArray(), keyp4, desc4, false);
	  }
    if(keyp1.size() > 6 || keyp2.size() > 6) //Segmentation fault if detected features are lower than 5
    {
    matchesOdom = BruteForce(oldCrop, crop, keyp1, keyp2, desc1, desc2, 0.5);
		matches2 = BruteForce(oldScaleIm, scaleIm, keyp3, keyp4, desc3, desc4, 0.5);

		//For triangulation, the matching 2D-correspondences from both frames are collected
		tie(scene1, scene2) = getPixLoc(keyp1, keyp2, matchesOdom);
		//Same for the smaller frame if we want to revocer scale as well
		if(scaleRecoveryMode)
		{
		tie(scene3, scene4) = getPixLoc(keyp3, keyp4, matches2);
 	  }

		//Initial epipolar geometry between the 2 current frames
    tie(t,R) = getInitPose(keyp1, keyp2, matchesOdom, K);

		Rodrigues(R, rotDiff, noArray());
		vector<Point3d> siftX;
	  vector<Point2d> sift1;
		vector<Point2d> sift2;
		avgMatchesDist = avgMatchDist(matchesOdom); //Average distance of

		//Where the scale recovery is obtained
		if(scaleRecoveryMode)
		{
		PkHat = scaleUpdate(Kscale, Rprev, Rprev*R, tprev, tprev + Rprev*t); //The scale
		cv::triangulatePoints(Kscale*Pk_1Hat, PkHat, scene3, scene4, point3d);
		curScale = getScale(point3d, scene4, PkHat, prevScale, alpha, camHeight);
		prevScale = curScale;
		cout << curScale << "," << scaleGT << ";" << endl;
	  }
		//velocity = curScale/correctTimeDivide(timeDiff, sampleTime); //Scale is the total displacement in meters, over the time of 100ms the velocity can be found
		//cout << "Total velocity: " << velocity << " m/s" << endl; //Display total velocity
		//cout << velocity << endl;
		if(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1 && avgMatchesDist > avgDistThresh)//Safeguard, image is still if avgMatches is higher than a threshold
		{
		Rodrigues(Rpos, rot, noArray()); 			//Converts rotation matrix to rotation vector
		Rodrigues(R, rotDiff, noArray());			//Same as above but with the
		yawDiff = rotDiff.at<double>(1,0);
		yaw = rot.at<double>(1,0); 						//Yaw for publishing odom message
		if(scaleRecoveryMode)
		{
			tpos = tpos + Rpos*t*curScale; 			//The scaled estimate is updated here
		}
		else
		{
		tpos = tpos + Rpos*t;					//The scaled estimate is updated here
	  }
		Rpos = R*Rpos;								//The rotation matrix updated after

		}
	}
		//cout << "Rot diff: " << rotDiff.at<double>(1,0)*180/3.14159 << endl;
		//cout << "t: " << t << endl;
		//cout << "Rot: " << Rpos << endl;
		//cout << endl;

		if(it > 1)
		{
    X = tpos.at<double>(0,0);
    Y = tpos.at<double>(0,2);
		x = tpos.at<double>(0,2);
		y = tpos.at<double>(0,0);
		vx = t.at<double>(0,2)*velocity;
		vy = t.at<double>(0,0)*velocity;
		yawVel = yawDiff/correctTimeDivide(timeDiff, sampleTime);

		//cout << "Xpos: " << x << endl;
		//cout << "Ypos: " << y << endl;
		//cout << "xvec: " << vx << endl;
		//cout << "yvec: " << vy << endl;
		//cout << "Yaw: " << yaw*180/3.14159 << endl;
		//cout << "Yaw difference: " << yawDiff*180/3.14159 << endl;
		//cout << x << "," << y << "," << -yGT << "," << xGT << "," << curScale << "," << scaleGT << "," << yawDiff*180/3.14159 << ";" << endl;
	  }
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
    //Publish the odometry message
    odom_pub_.publish(odom);

		circle(trajectory, Point(X + dim/2,-Y + dim/2), 1, Scalar(0,0,255), 1);
    cv::resize(trajectory, traj, cv::Size(), showScale, showScale);
    imshow("Trajectory", traj);
		//Copying previous parameters such as keypoints, descriptors etc
		oldCrop = crop.clone();
  	oldIm = Im.clone();
    keyp1 = keyp2;
    desc1 = desc2.clone();
		if(scaleRecoveryMode)
		{
		oldScaleIm = scaleIm.clone();
    keyp3 = keyp4;
    desc3 = desc4.clone();
	  }
		//Saving the previous epipolar geometry, mainly for the scale recovery
		Rprev = R.clone();
		tprev = t.clone();


    it++; //Iteration number
		//cout << "Frame number: " << it << endl;
		lastTime = simTime;
    //------------------------------------------------------------------------------------------------------
//-------------------------------------------------------------------------------------------------------------------
    // Update GUI Window
    //cv::imshow(OPENCV_WINDOW, frame);
    cv::waitKey(1);

  }
private:
	//Nodehandles
  ros::NodeHandle nh_;  //Image nodehandle
	ros::NodeHandle n;	  //Odometry nodehandle

	image_transport::Subscriber info_sub_;
	ros::Publisher odom_pub_ = n.advertise<nav_msgs::Odometry>("odom", 1);
	nav_msgs::Odometry odom;
  image_transport::ImageTransport it_;
  image_transport::Subscriber image_sub_;
};

int main(int argc, char** argv)
{
  ros::init(argc, argv, "image_converter");
	ros::NodeHandle nTf; //Ground truth nodehandle
	ros::Subscriber tf_msg = nTf.subscribe("/tf", 50, tfCb); //Ground truth message
  ImageConverter ic;
  ros::spin();
  return 0;
}
