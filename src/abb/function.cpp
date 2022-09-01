#include "function.h"
using namespace cv;
using namespace std;
//using namespace cv::xfeatures2d;
//
//Brute Force Matcher + displays matches
//
vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio) {
//Makes sure descriptors are the right data type
	if(desc1.type()!=CV_32F) {
	    desc1.convertTo(desc1, CV_32F);
			//cout << "Converting to CV_32F" << endl;
	}
	if(desc2.type()!=CV_32F) {
	    desc2.convertTo(desc2, CV_32F);
			//cout << "Converting to CV_32F" << endl;
	}
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_L2, false);
	std::vector< std::vector<DMatch> > matches;
	std::vector<DMatch> good_matches;
  if (keyp1.size() > 3 && keyp2.size() > 3) { //Minimum amount of features needed from both frames
	//Extract feature matches
	matcher->knnMatch(desc1, desc2, matches, 2);
	//Removes bad matches using the Lowe's ratio test
const float ratio_thresh = ratio;
for (size_t i = 0; i < matches.size(); i++)
{
		if (matches[i][0].distance < ratio_thresh * matches[i][1].distance)
		{
				good_matches.push_back(matches[i][0]);
		}
}
//Draw matches
Mat img_matches,img_matches_good;
drawMatches(img1, keyp1, img2, keyp2, good_matches, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
cv::resize(img_matches_good, img_matches_good, cv::Size(), 1, 1);
imshow("Good Matches", img_matches_good);
}
	return {good_matches};
}

//Returns epipolar geometry between 2 frames based on the RANSAC scheme
tuple <Mat, Mat> getInitPose(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches, Mat cam){
	vector<Point2f> scene1, scene2; //Pixel locations of feature matches
	Mat E, R, t; //Essential matrix, Rotation matrix and translation vector
if (matches.size() > 5) { //5-point algorithm won't work with less

for( size_t i = 0; i < matches.size(); i++)
{
		//Retrieve the Pixel locations from the good matches
		scene1.push_back( keyp1[ matches[i].queryIdx ].pt);
		scene2.push_back( keyp2[ matches[i].trainIdx ].pt);
}

//Estimating the epipolar geometry between the 2 frames with a RANSAC scheme
E = findEssentialMat(scene2, scene1, cam, RANSAC, 0.999, 1);
recoverPose(E, scene2, scene1, cam, R, t, noArray());
}
return {t,R};
}

//Not currently used in this subset
//Bucketing: Divides frame into a nxm grid, detects features in each subset
//I.e. evenly distributes the keypoints more evenly
vector<KeyPoint> Bucketing(Mat img, int gridX, int gridY, int max)
{
	//The ORB-detector
	Ptr<ORB> ORBucket = cv::ORB::create(
		  120, // Max 120 features per grid point
			1.2f,//Pyramid decimation ratio
			8,	 //Nr of pyramid levels
			31,
			0,
			2,
			ORB::FAST_SCORE,
			31,
			5); //Detector
Mat crop;
vector <KeyPoint> helpKeyp;
//Divides image into grid
int gridRows = img.cols/gridX;
int gridCols = img.rows/gridY;
vector <KeyPoint> keyp;
//Detects features for each grid subset
for(size_t i = 0; i < gridX; i++)
{
	for(size_t j = 0; j < gridY; j++)
	{
		cv::Rect crop_region((i)*gridRows, (j)*gridCols, gridRows, gridCols);
		crop = img(crop_region);
		cout << "At point: " << (i)*gridRows << " , " << (j)*gridCols << endl;
		ORBucket->detect(crop, helpKeyp, Mat());
		//cout << "Number of keypoints generated: " << helpKeyp.size() << endl;
		//cout << endl;
		for(size_t k = 1; k < helpKeyp.size(); k++)
		{
			helpKeyp[k].pt.x = helpKeyp[k].pt.x + (i)*gridRows;
			helpKeyp[k].pt.y = helpKeyp[k].pt.y + (j)*gridCols;
		}
		//keyp.push_back(helpKeyp);
		keyp.insert(keyp.end(), helpKeyp.begin(), helpKeyp.end());
	}
}
//drawKeypoints(img, keyp, img);
//cv::imshow("Bucketing output", img);
cout << "Number of keypoints generated: " << keyp.size() << endl;
return keyp;
}
//
//// Scale Update
//
Mat scaleUpdate(Mat K, Mat R_, Mat R, Mat t_, Mat t){

//Mat R0 = cv::Mat::eye(3,3,CV_64F);
Mat A = R*R_.inv();
Mat B = t - R*R_.inv()*t_;
Mat proj;
hconcat(A,B,proj);
Mat PkHat = K*proj;

return proj;
}

Mat projMat(Mat K, Mat R, Mat t)
{
	Mat proj;
	hconcat(R,t,proj);
	return K*proj;
}

tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches){
	vector<Point2d> scene1, scene2;
for( size_t i = 0; i < matches.size(); i++)
{
		//-- Retrieve the keypoints from the good matches
		scene1.push_back( keyp1[ matches[i].queryIdx ].pt);
		scene2.push_back( keyp2[ matches[i].trainIdx ].pt);
		//cout << keyp1[ matches[i].queryIdx ].pt << " " << keyp2[ matches[i].trainIdx ].pt << endl;
}
return{scene1, scene2};
}

double avg(vector<double> v)
{
        double return_value = 0.0;
        int n = v.size();
        for(int i = 0;i < n; i++)
        {
          return_value += v[i];
        }
        return return_value/n;
}
//Function for variance, help function to getScale
double variance(vector<double> v,double mean)
{
        double sum = 0.0;
        double temp = 0.0;
        double var = 0.0;
				auto length = v.size();
        for (int j = 0; j < length; j++)
        {
            temp = pow(v[j] - mean,2);
            sum += temp;
        }
        var = sum/(length);
				return var;
}

//The main algorithm for scale recovery
double getScale(Mat point3d,vector<Point2d> scene, Mat PkHat, double prevScale, double alpha, double height){
int size;
double outlierFactor = 2; //Factor to remove outliers from the median value with repr. error lower than maxError
vector<double> Yval, YvalFiltered;
Mat E; //Reprojection error vector
double maxError = 1000; //Max repr. error
double X,Y,Z,W, x,y, median, avgY, newScale, e;
	for(int i = 0; i<point3d.cols; i++){
		//Obtain Triangulated 3D-points and their 2D correspondence from last frame
		W = point3d.at<double>(i,3);
		X = point3d.at<double>(i,0)/W;
		Y = point3d.at<double>(i,1)/W;
		Z = point3d.at<double>(i,2)/W;
		x = scene[i].x;
		y = scene[i].y;

		double Xj[4][1] = {X, Y, Z, 1};
		double xj[3][1] = {x, y, 1};
		Mat XJ = Mat(4,1,CV_64F,Xj);
		Mat xJ = Mat(3,1,CV_64F,xj);
		//Calculate repr. error
		E = xJ - PkHat*XJ;
		e = sqrt(E.at<double>(0,0)*E.at<double>(0,0) + E.at<double>(0,1)*E.at<double>(0,1) + E.at<double>(0,2)*E.at<double>(0,2));
		//Some values of Y are close to 1 which are erroneous, therefore removed in condition below (if abs(Y-1) > 0.001)
		if(e < maxError && height/Y < outlierFactor*prevScale && height/Y > prevScale/outlierFactor && abs(Y-1) > 0.001)
		{
			//cout << "Reprojection error: " << e << endl;
			//cout << "Ok value: " << 1.65/Y << endl;
			//cout << pix << endl;
			//cout << endl;
			Yval.push_back(Y);
		}
		}
size = Yval.size();
//cout << "Size: " << size << endl;
sort(Yval.begin(), Yval.end());
for(int i = 0; i < size; i++){
//cout << 1.65/Yval[i] << endl;
}
if(size > 2)
{
if(size % 2 == 0) //Size of valid Y-points are even
{
	median = (Yval[size/2-1] + Yval[size/2])/2;
}
if(size % 2 == 1) //Size of valid Y-points are odd
{
	median = Yval[Yval.size()/2];
}
for(int i = 0; i < size; i++){
	//Remove last outliers
	if(Yval[i] < outlierFactor*median && Yval[i] > median/outlierFactor)
	{
		YvalFiltered.push_back(Yval[i]);
	}
}
avgY = avg(YvalFiltered);
}
else
{
	avgY = avg(Yval);
}
if(size == 0)
{
	avgY = height/prevScale; //Return previous scale if no valid points are found
}
//Scale smoothing, no smoothing if input alpha = 1
newScale = (1 - alpha)*prevScale + alpha*height/avgY;
return newScale;
}

double avgMatchDist(vector<DMatch> matches)
{
double avg;
double sum = 0;
	for (size_t i = 0; i < matches.size(); i++)
	{
		sum = sum + matches[i].distance;
  }
	avg = sum/matches.size();
	return avg;
}


double correctTimeDivide(double timeDiff, double sampleTime)
{
if(timeDiff < 1)
	{
		return round(10*timeDiff)*sampleTime;
	}
else
	{
		return sampleTime;
	}
}

// Converts a rotation vector to rotation matrix
Mat eulerAnglesToRotationMatrix(Mat rot)
	{
	    // Calculate rotation about x axis
	    Mat R_x = (Mat_<double>(3,3) <<
	               1,       0,              0,
	               0,       cos(rot.at<int>(0,0)),   -sin(rot.at<int>(0,0)),
	               0,       sin(rot.at<int>(0,0)),   cos(rot.at<int>(0,0))
	               );
	    // Calculate rotation about y axis
	    Mat R_y = (Mat_<double>(3,3) <<
	               cos(rot.at<int>(0,1)),    0,      sin(rot.at<int>(0,1)),
	               0,               1,      0,
	               -sin(rot.at<int>(0,1)),   0,      cos(rot.at<int>(0,1))
	               );
	    // Calculate rotation about z axis
	    Mat R_z = (Mat_<double>(3,3) <<
	               cos(rot.at<int>(0,2)),    -sin(rot.at<int>(0,2)),      0,
	               sin(rot.at<int>(0,2)),    cos(rot.at<int>(0,2)),       0,
	               0,               0,                  1);
	    Mat R = R_z * R_y * R_x;
	    return R;
	}

// Sifts out abnormaly large triangulated 3D-points and their 2D-point correspondences
tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>> siftPoints(Mat X3D, vector<Point2d> scene1, vector<Point2d> scene2)
{
vector<Point2d> sift1;
vector<Point2d> sift2;
double Xs,Ys,Zs;
int j = 0;
vector<Point3d> coords;
int limit = 1000;
for(int i = 0; i < X3D.cols; i++){
		Xs = X3D.at<double>(i,0)/X3D.at<double>(i,3);
		Ys = X3D.at<double>(i,1)/X3D.at<double>(i,3);
		Zs = X3D.at<double>(i,2)/X3D.at<double>(i,3);

	if(abs(Xs) < limit && abs(Ys) < limit && abs(Zs) < limit) // This condition sifts out
	{
		coords.push_back({Xs,Ys,Zs});
		sift1.push_back(scene1[i]);
		sift2.push_back(scene2[i]);
		j++;
	}
}
return{coords, sift1, sift2};
}

// Returns the index of the triangulated 3D-point with the lowest reprojection error
// Not currently used in the algorithm
int poseRefiner(Mat K, Mat P, vector<Point3d> Xtriang, vector<Point2d> scene1, vector<Point2d> scene2)
{
// Repr.error = |xk - Xk*P|
Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F); //
Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
Mat E; // Reprojection error matrix xk - Xk*P
double e; // Resulting reprojection error
double min = 1000000; // Start with relative high value to compare repr.errors with
int index;
for (size_t i = 0; i < Xtriang.size(); i++)
{
// Form the current 3D-point from input with index i
Xk.at<double>(0,0) = Xtriang[i].x;
Xk.at<double>(0,1) = Xtriang[i].y;
Xk.at<double>(0,2) = Xtriang[i].z;
// Form the current Image-point from input with index i
xk.at<double>(0,0) = scene2[i].x;
xk.at<double>(0,1) = scene2[i].y;

E = xk - P*Xk;
e = sqrt(E.at<double>(0,0)*E.at<double>(0,0) + E.at<double>(0,1)*E.at<double>(0,1) + E.at<double>(0,2)*E.at<double>(0,2));

if (e < min)
{
	min = e;
	index = i;
}
}
return index;
}

//
//// SUACE - Speeded Up Adaptive Constrast Enhancement filter
// This function is not utilized by the current algorithm
void performSUACE(Mat & src, Mat & dst, int distance, double sigma) {

	CV_Assert(src.type() == CV_8UC1);
	dst = Mat(src.size(), CV_8UC1);
	Mat smoothed;
	int val;
	int a, b;
	int adjuster;
	int half_distance = distance / 2;
	double distance_d = distance;

	GaussianBlur(src, smoothed, cv::Size(0, 0), sigma);

	for (int x = 0;x<src.cols;x++)
		for (int y = 0;y < src.rows;y++) {
			val = src.at<uchar>(y, x);
			adjuster = smoothed.at<uchar>(y, x);
			if ((val - adjuster) > distance_d)adjuster += (val - adjuster)*0.5;
			adjuster = adjuster < half_distance ? half_distance : adjuster;
			b = adjuster + half_distance;
			b = b > 255 ? 255 : b;
			a = b - distance;
			a = a < 0 ? 0 : a;

			if (val >= a && val <= b)
			{
				dst.at<uchar>(y, x) = (int)(((val - a) / distance_d) * 255);
			}
			else if (val < a) {
				dst.at<uchar>(y, x) = 0;
			}
			else if (val > b) {
				dst.at<uchar>(y, x) = 255;
			}
		}
}
