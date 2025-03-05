//Written by Alexander Smith, 2022-2023

#ifndef FUNCTION_H    // To make sure you don't declare the function more than once by including the header multiple times.
#define FUNCTION_H

#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/core/mat.hpp>
#include <chrono>  //For high_resolution_clock (Testing phase)
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

void showInliersOutliers(vector<size_t> index, vector<Point2d> scene,const Mat& Image, int xOffset, int yOffset, int iteration);

int binom(int n, int k);

double avg(vector<double> v);
double variance(vector<double> v,double mean);
double avgMatchDist(vector<DMatch> matches);

Mat normalizeVec(Mat tvec);
Mat rot2Quat(Mat rotvec);

tuple <Mat, Mat> getEpip(vector<Point2d> scen1, vector<Point2d> scen2, vector<DMatch> matches, Mat cam);
tuple <Mat, vector<KeyPoint>> detectFeatures(Mat img);
tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches);
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>,vector<size_t>> siftPoints(vector<Point3d> X3D, Mat proj, vector<Point2d> scene1, vector<Point2d> scene2, int maxPoints, int gmmOrder, bool show);
tuple <double, int> qScore(vector <double> heights);
tuple <vector<size_t>,vector<double>> inlierLikelihood(vector<double> yVal, int order);
tuple <Mat, Mat>projMatrices(Mat cam, Mat Rota, Mat tran);
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>> siftError(Mat proj, Mat Xtriang, vector<Point2d> scene1, vector<Point2d> scene2, int maxError);
tuple <double, Mat, vector<double>, double, double, double> generateHeights(vector<Point3d> X, double realH, double maxScale, int maxIter);
tuple <float,float,float> Quat2Euler(float q0, float q1, float q2, float q3);

vector<KeyPoint> shiTomasiHelp(Mat img,
															 int maxCorners,
															 double qualityLevel,
															 int minDistance,
															 int blockSize,
															 bool useHarrisDetector,
														   double k);
vector<KeyPoint> shiTomasi(Mat img);
vector<Point3d> dim4to3(Mat dim4);
vector<int> kthCombination(int n, int k, int m);
vector<int> randList(int listsize, int max);
vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio, bool hamm);



//
////Not used in the current algorithm
//
int randnr(int min, int max);
vector<vector<int>> combList(int n, int k);
void nelderMead(Point3d normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota);
void nelder(Mat normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota);
void dispProjError(Mat Proj, vector<Point2d> pt2, vector<Point3d> p3d);
void spherical(Point3d normvec);
void PermGenerator(int n, int k);
void orbDraw(Mat img);
void testHomog(vector<double> hestimates, vector<Point2d> scene2, Mat cam, vector<Point3d> Xground);
void inliers(vector<Point3d> X, int order);
vector<double> removeLoners(vector<double> val, double dist);
#endif
