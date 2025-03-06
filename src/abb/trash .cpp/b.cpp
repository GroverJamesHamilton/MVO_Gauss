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
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
//Own-made functions from function.cpp
#include "function.h"
//For tf, ground truth data in KITTI datasets are stored here
#include <tf/transform_listener.h>
#include "tf/message_filter.h"
#include "message_filters/subscriber.h"
#include <tf2_ros/transform_listener.h>
#include <opencv2/viz.hpp>
//Algorithm iteration index
double it = 1;
//Frame parameters for different purposes
Mat oldIm, oldIm0, Im, Im0, ImS, oldImS;
//Below are the rotation and translation matrices used for different purposes
Mat R = cv::Mat::eye(3,3,CV_64F); // Rotation matrix from epipolar geometry between 2 frames
Mat Rpos = cv::Mat::eye(3,3,CV_64F); // Rotation matrix
Mat Rprev = cv::Mat::eye(3,3,CV_64F);
Mat tpos = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat rot = cv::Mat::eye(3,3,CV_64F);
Mat rotDiff = cv::Mat::eye(3,3,CV_64F);
Mat t = cv::Mat::zeros(cv::Size(1,3), CV_64F);
Mat tprev = cv::Mat::zeros(cv::Size(1,3), CV_64F);
//Transformed projection matrices, for the scale recovery
cv::Mat PkHat;
Mat Pk_1Hat = cv::Mat::eye(cv::Size(4,3), CV_64F);
//Features, descriptors, matches
vector<KeyPoint> keyp1, keyp2, keyp3, keyp4, keyp5, keyp6;
Mat desc1, desc2, desc3, desc4, desc5, desc6;
vector<DMatch> matchesOdom, matches2, matches3;
//Triangulated 3D-points
cv::Mat point3d, pointX;
vector<Point2d> scene1, scene2, scene3, scene4, scene5, scene6;
//For publishing the yaw parameters
float yaw, yawDiff, yawVel;
double sampleTime = 0.103; //The sample time for the rosbag dataset in ms, default would be 100 or 100/3
//Display parameters
double dim = 200;
double dimShow = 500;
double showScale = dimShow/dim;
//Scale recovery parameters
double curScale = 1;
double prevScale = 1;
double alpha = 1; //Scale smoothing parameters, alpha = [0 1], 1 is no smoothing
double camHeight = 1.65; //Camera height, for scale recovery
vector<Point3d> Xground;
vector<Point2d> sift1, sift2;
//Scale recovery frame cropping
int xpix = 1242;
int ypix = 375;

int xO2 = 450;
int yO2 = 225;
cv::Rect cropScale(xO2, yO2, xpix - 2*xO2, ypix - yO2); //Only a small subset of the frame is extracted
//The printed trajectory, Mainly for visually
Mat trajectory = Mat::zeros(dim, dim, CV_8UC3);
Mat traj;
int X,Y; //Pixel location to print trajectory
double x,y,vx,vy;
//Initialize the ORB detectors/descriptors
//For the unscaled visual odometry
//Primarily: The max numbers of features and FAST threshold are the most crucial parameters
//to change since they have the most impact
Ptr<ORB> orbOdom = cv::ORB::create(840, //Max number of features
		1.25f,															 //Pyramid decimation ratio > 1
		8,																	 //Number of pyramid levels
		16,																	 //Edge threshold
		0,																	 //The level of pyramid to put source image to
		3,																	 //WTA_K
		ORB::FAST_SCORE,										 //Which algorithm is used to rank features, either FAST_SCORE or HARRIS_SCORE
	  31,                                  //Descriptor patch size
		9);                                  //The FAST threshold

//For the scale recovery
Ptr<ORB> orbScale = cv::ORB::create(3000, 1.1f, 12, 5, 0, 2, ORB::HARRIS_SCORE, 40, 25);

double fx = 718.856; //903.7596
double fy = 718.856; //901.9653
double ox = 607.1928; //695.7519
double oy = 185.2157; //224.2509

//The camera matrix dist. coeffs for the KITTI dataset
 double kOdom[3][3] = {
	  {fx,0,ox}, //Principal point needs to be offset if image is cropped
	  {0,fy,oy},
	  {0,0,1}};

 double kScale[3][3] = {
		{fx,0,ox - xO2}, //Principal point needs to be offset if image is cropped
		{0,fy,oy - yO2},
		{0,0,1}};

	Mat K = Mat(3,3,CV_64F,kOdom);
	Mat Kscale = Mat(3,3,CV_64F,kScale);
  double distCoeffs[5][1] = {-0.3691481, 0.1968681, 0.001353473, 0.0005677587, -0.06770705};
  Mat dist = Mat(5,1,CV_64F,distCoeffs);
	Mat Pk_0 = cv::Mat::eye(cv::Size(4,3), CV_64F);
	Mat Pk_1 = Kscale*Pk_0;
	Mat Pk;

// Ros time stamp
using namespace cv;
using namespace std;
static const std::string OPENCV_WINDOW = "Image Window";

double xGT, yGT, zGT, xGT2, yGT2, scaleGT; //Ground truth parameters from node /tf
double yawGT, yawDiffGT;
double yawprevGT = -3.00251;
double velocity;

//Change parameters here:
bool scaleRecoveryMode = true;
int queueSize = 30;

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
auto yawDiffGT = yawGT - yawprevGT;//cout << yawDiffGT << endl;//Ground truth scale
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
															//Replace this with your topic below//
		image_sub_ = it_.subscribe("/kitti/camera_gray_left/image_raw", queueSize, &ImageConverter::imageCb, this);
    cv::namedWindow(OPENCV_WINDOW);
  }
  ~ImageConverter()
  {
    cv::destroyWindow(OPENCV_WINDOW);
  }
	//Image callback, where the main algorithm is
  void imageCb(const sensor_msgs::ImageConstPtr& msg)
  {
    cv_bridge::CvImagePtr cv_ptr;
		cv_bridge::CvImagePtr img_ptr;
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
		//imp
    oldIm = cv_ptr->image; //Receive image
		//keyp1 = shiTomasiHelp(oldIm, 2000, 0.01, 10, 3, false, 0.04);
		//orbOdom->compute(oldIm, keyp1, desc1);
		orbOdom->detectAndCompute(oldIm, noArray(), keyp1, desc1, false);
		//Same with a subset of the frame if the scale recovery is wanted
		if(scaleRecoveryMode)
		{
		oldImS = oldIm(cropScale);
		//orbScale->detectAndCompute(oldImS, noArray(), keyp5, desc5, false);
		keyp5 = shiTomasiHelp(oldImS, 1000, 0.05, 3, 3, false, 0.04);
		orbScale->compute(oldImS, keyp5, desc5);
	  }
  }
    Im = cv_ptr->image;
		//keyp2 = shiTomasiHelp(Im, 2000, 0.01, 10, 3, false, 0.04);
		//orbOdom->compute(Im, keyp2, desc2);
		orbOdom->detectAndCompute(Im, noArray(), keyp2, desc2, false);
		if(scaleRecoveryMode)
		{
		ImS = oldIm(cropScale);
		//orbScale->detectAndCompute(ImS, noArray(), keyp6, desc6, false);
		keyp6 = shiTomasiHelp(ImS, 1000, 0.05, 3, 3, false, 0.04);
		orbScale->compute(ImS, keyp6, desc6);
	  }
    if(keyp1.size() > 6 || keyp2.size() > 6) //Segmentation fault if detected features are lower than 5
    {
    matchesOdom = BruteForce(oldIm, Im, keyp1, keyp2, desc1, desc2, 0.75); //0.75
		matches3 = BruteForce(oldImS, ImS, keyp5, keyp6, desc5, desc6, 0.75); //
		//For triangulation, the matching 2D-correspondences from both frames are collected
		tie(scene1, scene2) = getPixLoc(keyp1, keyp2, matchesOdom);
		//Same for the smaller frame if we want to recover scale as well
		if(scaleRecoveryMode)
		{
		tie(scene5, scene6) = getPixLoc(keyp5, keyp6, matches3);
 	  }
		//Initial epipolar geometry between the 2 current frames
    tie(t,R) = getInitPose(keyp1, keyp2, matchesOdom, K);
		Rodrigues(R, rotDiff, noArray());

			if(it > 2)
			 //if(it == 10)
			{
				//cout << "Keyp5: " << keyp5.size() << " Keyp6: " << keyp6.size() << endl;
				//cout << "It: " << it << endl;
				Mat M = (Mat_<double>(3,4) << 1.0, 0.0, 0.0, 0.0,
                              				0.0, 1.0, 0.0, 0.0,
                              				0.0, 0.0, 1.0, 0.0);
				Mat P1 = Kscale*M;
				cv::vconcat(R,t.t(),M);
				Mat P2 = Kscale*M.t();
				Mat pnts3D(4, scene5.size(), CV_64F);
				Mat vis(3, scene5.size(), CV_64F);
			  //cout << "Scene size: " << scene5.size() << endl;
				cv::triangulatePoints(P1, P2, scene5, scene6, pnts3D);
				//Xground = dim4to3(pnts3D);
				tie(Xground, sift1, sift2) = siftPoints(pnts3D, P2, scene5, scene6, 14, 5000, false);
				//cout << "Size: " << sift1.size() << endl;
				vector<Point3d> normalVec;
				double n1,n2,n3,hprim;
				generateHeights(Xground, 1.65, 10);
				//cout << "Rot.diff: " << rotDiff*180/3.14159 << endl;
				//cout << "Tran: " << t << endl;
				cout << scaleGT << " ;" << endl;
				double norm = sqrt(pow(xGT, 2) + pow(yGT, 2) + pow(zGT, 2));
				cout << endl; /*
			//tie(hprim,n1,n2,n3) = generateRand(Xground, 50000, sift1);
			//normalVec.push_back({n1,n2,n3});
			//nelderMead(normalVec, hprim, Kscale, sift1, sift2, t, R);
			//cout << "Normal vector: " << n1 << "," << n2 << "," << n3 << endl;
			//cout << "Estimated height: " << hprim << endl;
			*/
			Mat canvas = Mat::zeros(150, 290, CV_8UC3);
			for(int i = 0; i < sift1.size(); i++)
			{
				std::string s = std::to_string(i);
				putText(canvas,s,Point(sift1[i].x,sift1[i].y), cv::FONT_HERSHEY_DUPLEX,0.3,Scalar(0,0,255),1);
				putText(canvas,s,Point(sift2[i].x,sift2[i].y), cv::FONT_HERSHEY_DUPLEX,0.3,Scalar(0,255,0),1);
			}
			cv::resize(canvas, canvas, cv::Size(), 3, 3);
			imshow("Canvas", canvas);
		}
		//Where the scale recovery is obtained
		if(scaleRecoveryMode && it == 100)
		{
			Mat M = (Mat_<double>(3,4) << 1.0, 0.0, 0.0, 0.0,
																		0.0, 1.0, 0.0, 0.0,
																		0.0, 0.0, 1.0, 0.0);
			Mat P1 = Kscale*M;
		cout << "XXXXXXXXXXXXXXXXXXXXXXXXXX" << endl;
		PkHat = scaleUpdate(Kscale, Rprev, R, tprev, t); //The scale
		//PkHat = scaleUpdate(Kscale, Rprev, Rprev*R, tprev, tprev + Rprev*t); //The scale
		cv::triangulatePoints(P1, PkHat, scene5, scene6, point3d);
		curScale = getScale(point3d, scene5, PkHat, prevScale, alpha, camHeight);
		//prevScale = curScale;
		//cout << curScale << "," << scaleGT << ";" << endl;
	  }
		Rodrigues(Rpos, rot, noArray()); 			//Converts rotation matrix to rotation vector
		Rodrigues(R, rotDiff, noArray());			//Same as above but with the
		yawDiff = rotDiff.at<double>(1,0);
		yaw = rot.at<double>(1,0); 						//Yaw for publishing odom message
		if(R.rows == 3 && R.cols == 3 && t.rows == 3 && t.cols == 1 && abs(yawDiff*180/3.14159) < 30)
		{
		if(scaleRecoveryMode && it > 2)
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
		if(it > 1)
		{
    X = tpos.at<double>(0,0);
    Y = tpos.at<double>(0,2);
		/*
		x = tpos.at<double>(0,2);
		y = tpos.at<double>(0,0);
		vx = t.at<double>(0,2)*velocity;
		vy = t.at<double>(0,0)*velocity;
		*/
	  }
		//Quaternion created from yaw to publish in nav_msgs
		/*
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
    //set the position
    odom.pose.pose.position.x = x;
    odom.pose.pose.position.y = y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation = odom_quat;
		odom.twist.twist.linear.x = vx;
		odom.twist.twist.linear.y = vy;
		odom.twist.twist.angular.z = yawVel;
    //Publish the odometry message
    odom_pub_.publish(odom);
		*/
		circle(trajectory, Point(X + dim/2,-Y + dim/2), 1, Scalar(0,0,255), 1);
    cv::resize(trajectory, traj, cv::Size(), showScale, showScale);
    imshow("Trajectory", traj);
		//Copying previous parameters such as keypoints, descriptors etc
  	oldIm = Im.clone();
    keyp1 = keyp2;
    desc1 = desc2.clone();
		if(scaleRecoveryMode && it > 2)
		{
		oldImS = ImS.clone();
    keyp3 = keyp4;
    desc3 = desc4.clone();
		keyp5 = keyp6;
    desc5 = desc6.clone();
	  }
		//Saving the previous epipolar geometry, mainly for the scale recovery
		Rprev = R.clone();
		tprev = t.clone();

    it++; //Iteration number
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
	ros::Subscriber tf_msg = nTf.subscribe("/tf", queueSize, tfCb); //Ground truth message
  ImageConverter ic;
  ros::spin();
  return 0;
}
