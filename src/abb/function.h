#ifndef FUNCTION_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define FUNCTION_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <chrono>  // for high_resolution_clock
#include <tuple>
#include <vector>
#include <cstdlib>
#include <unistd.h>
#include <ros/ros.h>
#include <opencv2/viz.hpp>

#include <random>
#include <algorithm>
#include <iterator>

#include <armadillo>

using namespace cv;
using namespace std;

vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio);
vector<DMatch> Brute(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio);
tuple <Mat, Mat> getInitPose(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches, Mat cam);
tuple <Mat, Mat> displacement(Mat img1, Mat img2);

tuple <Mat, vector<KeyPoint>> detectFeatures(Mat img);
vector<KeyPoint> Bucketing(Mat img, int gridX, int gridY, int max);
bool isRotationMatrix(Mat &R);
tuple <float, float, float> rotationMatrixToEulerAngles(Mat &R);

Mat scaleUpdate(Mat K, Mat R_, Mat R, Mat t_, Mat t);
tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches);
double getScale(Mat point3d, vector<Point2d> scene, Mat PkHat, double prevScale, double alpha, double height);
double avg(vector<double> v);
double variance(vector<double> v,double mean);
double avgMatchDist(vector<DMatch> matches);
double correctTimeDivide(double timeDiff, double sampleTime);
Mat projMat(Mat cam, Mat rota, Mat t);
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>> siftPoints(Mat X3D, Mat proj, vector<Point2d> scene1, vector<Point2d> scene2, int maxPoints, int maxError, bool show);
Mat eulerAnglesToRotationMatrix(Mat rot);
int poseRefiner(Mat cam, Mat proj, vector<Point3d> Xtriang, vector<Point2d> scene1, vector<Point2d> scene2);
void disp(Mat Xg);
int binom(int n, int k);
tuple <double, int> qScore(vector <double> heights);
void medianAvg(vector <double> heights);
void nelderMead(vector<Point3d> normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota);
vector<size_t> inlierLikelihood(vector<double> yVal, double maxSigma);
vector<KeyPoint> shiTomasi(Mat img);
void orbDraw(Mat img);
void PermGenerator(int n, int k);
vector <vector <int>> combList(int n, int k);
tuple <double, double, double, double> generateHeights(vector<Point3d> X, double realH, double maxScale);
vector<KeyPoint> shiTomasiHelp(Mat img,
															 int maxCorners,
															 double qualityLevel,
															 int minDistance,
															 int blockSize,
															 bool useHarrisDetector,
														   double k);
vector<Point3d> dim4to3(Mat dim4);
void dispProjError(Mat Proj, vector<Point2d> pt2, vector<Point3d> p3d);
tuple <Mat, Mat> refinePose(Mat cam, vector<Point2d> pt1, vector<Point2d> pt2, Mat Rt, Mat tt);
Mat normalizeVec(Mat tvec);

#endif
