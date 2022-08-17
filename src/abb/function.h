#ifndef FUNCTION_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define FUNCTION_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
//#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <chrono>  // for high_resolution_clock
#include <tuple>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <ros/ros.h>

using namespace cv;
using namespace std;

// Return the image circled with the keypoints, the calculation time
// and the amount of keypoints
Mat ORB_(Mat image, vector<KeyPoint> keypoints);
Mat BRIEF(Mat image, vector<KeyPoint> keypoints);
Mat FLANN(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio);
vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio);
tuple <Mat, Mat> tranRot(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches);
tuple <Mat, Mat> displacement(Mat img1, Mat img2);
Mat zhangCalib(vector<string> fileNames, int cols, int rows);

tuple <Mat, vector<KeyPoint>> detectFeatures(Mat img);

// New functions
vector<KeyPoint> Bucketing(Mat img, int gridX, int gridY, int max);
bool isRotationMatrix(Mat &R);
tuple <float, float, float> rotationMatrixToEulerAngles(Mat &R);

void performSUACE(Mat & src, Mat & dst, int distance, double sigma);
Mat scaleUpdate(Mat K, Mat R_, Mat R, Mat t_, Mat t);
tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches);
double getScale(Mat point3d, Mat PkHat, vector<DMatch> matches, vector<KeyPoint> keyp2, double prevScale, double alpha, double height);
double avg(vector<double> v);
double variance(vector<double> v,double mean);
double avgMatchDist(vector<DMatch> matches);
double correctTimeDivide(double timeDiff, double sampleTime);

#endif
