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


vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio);
tuple <Mat, Mat> getInitPose(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches, Mat cam);
tuple <Mat, Mat> displacement(Mat img1, Mat img2);

tuple <Mat, vector<KeyPoint>> detectFeatures(Mat img);
vector<KeyPoint> Bucketing(Mat img, int gridX, int gridY, int max);
bool isRotationMatrix(Mat &R);
tuple <float, float, float> rotationMatrixToEulerAngles(Mat &R);

void performSUACE(Mat & src, Mat & dst, int distance, double sigma);
Mat scaleUpdate(Mat K, Mat R_, Mat R, Mat t_, Mat t);
tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches);
double getScale(Mat point3d, vector<Point2d> scene, Mat PkHat, double prevScale, double alpha, double height);
double avg(vector<double> v);
double variance(vector<double> v,double mean);
double avgMatchDist(vector<DMatch> matches);
double correctTimeDivide(double timeDiff, double sampleTime);
Mat projMat(Mat K, Mat R, Mat t);
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>> siftPoints(Mat X3D, vector<Point2d> scene1, vector<Point2d> scene2);
Mat eulerAnglesToRotationMatrix(Mat rot);
int poseRefiner(Mat K, Mat P, vector<Point3d> Xtriang, vector<Point2d> scene1, vector<Point2d> scene2);
#endif
