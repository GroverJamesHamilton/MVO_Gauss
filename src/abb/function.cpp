//Written by Alexander Smith, 2022-2023

#include "function.h"
using namespace cv;
using namespace std;
//
//Brute Force Matcher
//
vector<DMatch> BruteForce(Mat img1, Mat img2, vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, Mat desc1, Mat desc2, double ratio, bool hamm) {
	Ptr<BFMatcher> matcher = BFMatcher::create(NORM_HAMMING2, false);
	if(hamm)//Change to regular Hamming distance is hamm
	{
		matcher = BFMatcher::create(NORM_HAMMING, false);
	}
	vector<vector<DMatch>> matches;
	vector<DMatch> good_matches;
  if (keyp1.size() > 3 && keyp2.size() > 3) { //Minimum amount of features needed from both frames
	//Extract feature matches
	matcher->knnMatch(desc1, desc2, matches, 2);
	//Removes bad matches using the Lowe's ratio test
for(size_t i = 0; i < matches.size(); i++)
{
		if(matches[i][0].distance < ratio*matches[i][1].distance)
		{
				good_matches.push_back(matches[i][0]);//Save match as inlier if Lowe's condition is met
		}
}
//Draw matches
//Mat img_matches,img_matches_good;
//drawMatches(img1, keyp1, img2, keyp2, good_matches, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
//cv::resize(img_matches_good, img_matches_good, cv::Size(), 2, 2);
//imshow("Good Matches", img_matches_good);
}
	return {good_matches};
}

//Converts keypoint as data type into 2D-correspondences
tuple <vector<Point2d>, vector<Point2d>> getPixLoc(vector<KeyPoint> keyp1, vector<KeyPoint> keyp2, vector<DMatch> matches){
vector<Point2d> scene1, scene2;
for(size_t i = 0; i < matches.size(); i++)
{
		//Retrieve the keypoints from the good matches in pixel coordinates
		scene1.push_back(keyp1[ matches[i].queryIdx ].pt);
		scene2.push_back(keyp2[ matches[i].trainIdx ].pt);
}
return{scene1, scene2};
}

//Returns epipolar geometry between 2 frames based on the RANSAC scheme
tuple <Mat, Mat> getEpip(vector<Point2d> scen1, vector<Point2d> scen2, vector<DMatch> matches, Mat cam){
	vector<Point2f> scene1, scene2; //Pixel locations of feature matches
	Mat Em, Rm, t; //Essential matrix, Rotation matrix and translation vector
if (matches.size() > 5) { //5-point algorithm won't work with less

//Estimating the epipolar geometry between the 2 frames with a RANSAC scheme
Em = findEssentialMat(scen2, scen1, cam, RANSAC, 0.995, 1);
recoverPose(Em, scen2, scen1, cam, Rm, t, noArray());//||t|| = 1 (Arbitrary scale factor)
}
return {t,Rm};
}

double avg(vector<double> v)
{
        double sumVal = 0;
        for(uint16_t i = 0; i < v.size(); i++)
        {
          sumVal += v[i];
        }
        return sumVal/v.size();
}

double variance(vector<double> v,double mean)
{
	double sum = 0;
	double temp = 0;
	for (uint16_t j = 0; j < v.size(); j++)
	{
		temp = pow(v[j] - mean,2);
		sum += temp;
	}
	return sum/v.size();
}

double avgMatchDist(vector<DMatch> matches)
{
double sum = 0;
	for (size_t i = 0; i < matches.size(); i++)
	{
		sum += matches[i].distance;
  }
	return sum/matches.size();
}

//Trains 1-dimensional GMM-models for different orders, sets the points
//as an inlier if in the IQR-range of gaussian with the lowest mean,
//given that it's weight is significant enough
//Returns index of which 3D-points are inliers/ground points and their likelihood
tuple<vector<size_t>, vector<double>> inlierLikelihood(vector<double> yVal, int order)
	{
		double minimumweight = 0.1;//The minimum weight
		double minw = 1;
		double bic, aic, aicc;
		bool status;
		int minIdx;
		double pi = 3.14159;
		double minScore = 10000;
		double mu = avg(yVal);
		double var = variance(yVal, mu);
		double sigma = sqrt(var);
		int N = yVal.size();
 		arma::gmm_diag model;
		arma::mat data(1, yVal.size(), arma::fill::zeros);
		int k;
		for(uint16_t i = 0; i < yVal.size(); i++)
		{
			data(0,i) = yVal[i];
		}
		if(yVal.size() > 10)
		{
			for(size_t k = 1; k <= order; k++)//Training models
			{
			status = model.learn(data, k, arma::eucl_dist, arma::random_subset, 10, 5, 1e-10, false);
			if(status)//If training successful, find
			{
				vector<double> means;
				//Converts from model to vector, easier to handle
				for(uint16_t i = 0; i < k; i++){means.push_back(model.means(i));}
				//Finds the index of lowest mean
				auto elements = minmax_element(begin(means), end(means));
				minIdx = std::distance(means.begin(), elements.first);
				//sort(means.begin(), means.end());

				//Information criterions, determines the model fit
				aic = 2*k - 2*model.sum_log_p(data);
				aicc = aic + (2*k*(k + 1))/(N - k - 1);
				bic = k*log(N) - 2*model.sum_log_p(data);
				cout << "BIC: " << bic << endl;
				minw = model.hefts(minIdx);
				cout << "MinW: " << minw << endl;
				//Saves the lowest gaussian (mu_min,sigma_min) if the model is most fit
				//and the weight is significant
				if(bic < minScore && minw > minimumweight)
				{
					minScore = bic;
					mu = model.means[minIdx];
					sigma = sqrt(model.dcovs[minIdx]);
				}
			}
		}
			}
	//IQR parameters to exclude outliers from ground points
	double range = -0.6745;
	double Q1 = sigma*range + mu;
	double Q3 = -sigma*range + mu;
	double lh;
	vector<size_t> index;
	vector<double> lhoods;
	//Retrieves the likelihoods of the inliers,
	//this way other functions can take inliers with the highest likehood w.r.t. the lowest guassian
	for(uint16_t i = 0; i < yVal.size(); i++)
	{
		//If in IQR-range, save as inlier + likelihood
		if(yVal[i] > Q1 && yVal[i] < Q3)
		{
			lh = 1/(sqrt(2*pi)*sigma)*exp(-pow((yVal[i] - mu)/sigma, 2)/2);
			lhoods.push_back(lh);
			index.push_back(i);
		}
	}
	//Sort inliers with highest likelihood first,
	//other functions can then choose the N most likely candidates
	stable_sort(index.begin(), index.end(), [&lhoods](size_t i1, size_t i2) {return lhoods[i1] > lhoods[i2];});
	vector<double> likelihoods;
	return {index, lhoods};
}

//Sifts out the triangulated ground 3D-points and their 2D-point correspondences
//from the outliers based on their Y-coordinates
		tuple <vector<Point3d>,vector<Point2d>,vector<Point2d>,vector<size_t>> siftPoints(vector<Point3d> X3D, Mat proj, vector<Point2d> scene1, vector<Point2d> scene2, int maxPoints, int gmmOrder, bool show)
		{
		vector<double> yVal;
		vector<Point2d> sift1, sift2;
		vector<Point3d> coordsSifted;
		double Ys;
		vector<size_t> index;
		vector<double> likelihoods;
		//Extract the Y-values for GMM-training
		for(uint16_t i = 0; i < X3D.size(); i++)
		{
			Ys = X3D[i].y;
			yVal.push_back(Ys);
		}
		tie(index,likelihoods) = inlierLikelihood(yVal, gmmOrder);
		//Viz: Displays 3D-points, for frame analysis
		bool dispViz = 0;
		if(dispViz)
		{
		viz::Viz3d window("3D-coords");
		window.showWidget("coordinate", viz::WCoordinateSystem());
		window.showWidget("points", viz::WCloud(X3D, viz::Color::green()));
		window.spin();
		}
		if(maxPoints > index.size())
		{
			maxPoints = index.size();
		}
		for(uint16_t n = 0; n < maxPoints; n++)
		{
			if(n == yVal.size())
			{
				break;
			}
			sift1.push_back(scene1[index[n]]);
			sift2.push_back(scene2[index[n]]);
			coordsSifted.push_back(X3D[index[n]]);
		}
		//Returns the inliers sifted from
		return{coordsSifted, sift1, sift2, index};
		}

	//Print all the inliers (ground points) in the ROI to the full frame
	void showInliersOutliers(vector<size_t> index, vector<Point2d> scene, const Mat& Image, int xOffset, int yOffset, int iteration)
	{
		Mat imageCopy = Image.clone();//Copy Mat input, to not alter the original
																	//Const doesn't seem to keep the class unaltered outside of function
		int X,Y, Xend, Yend, sizeX, sizeY;
		std::vector<int> inliers(scene.size(), 0);
		for(uint16_t n = 0; n < index.size(); n++)
		{
			inliers[index[n]] = 1;
		}
		for(uint16_t i = 0; i < scene.size(); i++)
		{
			X = scene[i].x + xOffset;
			Y = scene[i].y + yOffset;
			if(inliers[i])
			{
				circle(imageCopy, Point(X,Y), 1, Scalar(124,252,0), 2);//Draws inlier(green)
			}
			else
			{
				circle(imageCopy, Point(X,Y), 1, Scalar(124,252,0), 2);//Draws inlier(green)
				//circle(imageCopy, Point(X,Y), 1, Scalar(0,0,255), 2);	//Draws outlier(red)
			}
		}
		string text = "Frame nr: ";         //Display frame number
		text += std::to_string(iteration);
		cv::putText(imageCopy, 						  //Target image
	            	text, 									//Text
	            	cv::Point(20, 20), 			//Top-left position
	            	cv::FONT_HERSHEY_DUPLEX,//Font
	            	0.5,
	            	CV_RGB(0,0,0), 					//Font color
	            	1);
		//Draws ROI bounding box
		rectangle(imageCopy, Point(xOffset,yOffset), Point(imageCopy.cols - xOffset,imageCopy.rows), Scalar(220,220,220), LINE_4, LINE_4);
		imshow("Inliers", imageCopy);
	}

//Returns projection matrices for triangulation
	tuple<Mat, Mat> projMatrices(Mat cam, Mat Rota, Mat tran)
	{
		Mat M = (Mat_<double>(3,4) << 1.0, 0.0, 0.0, 0.0,
																	0.0, 1.0, 0.0, 0.0,
																	0.0, 0.0, 1.0, 0.0);
		Mat proj1 = cam*M;
		cv::vconcat(Rota,tran.t(),M);
		Mat proj2 = cam*M.t();
		return{proj1, proj2};
	}

//Reject outliers w.r.t. reprojection error
	tuple<vector<Point3d>,vector<Point2d>,vector<Point2d>> siftError(Mat proj, Mat Xtriang, vector<Point2d> scene1, vector<Point2d> scene2, int maxError)
	{
		vector<Point3d> siftX; 												//The sifted 3D-points
		vector<Point2d> sift1, sift2;									//The sifted matching 2D-correspondences
		Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F);//Xk = [X,Y,Z,1]^T
		Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);//xk = [x,y,1]^T
		Mat Er; 																			//Reprojection error matrix xk - Xk*P
		double e; 																		//Resulting reprojection error
		for (size_t i = 0; i < Xtriang.cols; i++)
		{
			//[X*sc,Y*sc,Z*sc,sc]^T -> [X,Y,Z,1]^T
			Xk.at<double>(0,0) = Xtriang.at<double>(0,i)/Xtriang.at<double>(3,i);
			Xk.at<double>(0,1) = Xtriang.at<double>(1,i)/Xtriang.at<double>(3,i);
			Xk.at<double>(0,2) = Xtriang.at<double>(2,i)/Xtriang.at<double>(3,i);
			xk.at<double>(0,0) = scene2[i].x;
			xk.at<double>(0,1) = scene2[i].y;
			//Reprojection errors
			Er = xk - proj*Xk;
			e = sqrt(Er.at<double>(0,0)*Er.at<double>(0,0) + Er.at<double>(0,1)*Er.at<double>(0,1) + Er.at<double>(0,2)*Er.at<double>(0,2));

			if (e < maxError)
			{
				//Saves values if the reprojection error is lowe enough
				sift1.push_back(scene1[i]);
				sift2.push_back(scene2[i]);
				siftX.push_back({Xk.at<double>(0,0),Xk.at<double>(0,1),Xk.at<double>(0,2)});
				cout << Xk.at<double>(0,0) << "," << Xk.at<double>(0,1) << "," << Xk.at<double>(0,2) << ";" << endl;
			}
		}
		return{siftX, sift1, sift2};
	}

//Returns binomial of bin(n,k)
	int binom(int n, int k)
	{
	   if (k == 0 || k == n)return 1;
		 else if (n == 0)return 0;
		 else if (n == 0 && k == 0)return 1;
	   return binom(n-1, k-1) + binom(n-1, k);
	}

//Finds the most reasonable height from a set of estimated ground plane heights,
//See Equation 5.8 in paper "Real-Time Monocular Large-scale Multicore Visual Odometry, Shiyu Song" (Page 64)
//Returns the height that maximizes the score and its index from the list
	tuple <double, int> qScore(vector <double> heights)
	{
		double max = 0;
		double diff;
		double score;
		double diffSum;
		double mu = 50;
		int index;
		for(uint16_t i = 0; i < heights.size(); i++)
		{
			diffSum = 0;
		for(uint16_t j = 0; j < heights.size(); j++)
		{
			if(i != j)
			{
				diff = heights[i] - heights[j];
				score = exp(-mu*pow(diff,2));
				diffSum = diffSum + score;
			}
		}
	if(diffSum > max)//Save the height and index with maximum q-score
	{
		max = diffSum;
		index = i;
	}
	}
	return {heights[index], index};
	}

//Generate all heights hprim,
//the best estimated hprim relates the true scale/depth as scale = cameraHeight/hprim
	tuple <double, Mat, vector<double>, double, double, double> generateHeights(vector<Point3d> X, double realH, double maxScale, int maxIter)
	{
		Mat normVec = (Mat_<double>(1,3) << 0,-1,0);//Default normal vector
		vector<Point3d> normal;
		Mat point, N, N_norm;
		Mat n = cv::Mat::zeros(3,3,CV_64F);
		Mat l = cv::Mat::ones(3,1,CV_64F);
		double norm, height, n1, n2, n3, acceptratio;
		vector<double> heights;
		acceptratio = 0;
		vector<int> rand3;
		if(X.size() > 2)//Makes sure size is appropriate
		{
			int u = 0;
			for(uint16_t i = 0; i < maxIter; i++)
		{
		rand3 = randList(3,X.size());//Generate random unique sample from 1,...,X.size()
			for(uint16_t j = 0; j < 3; j++)
			{
				n.at<double>(j,0) = X[rand3.at(j)].x;
				n.at<double>(j,1) = X[rand3.at(j)].y;
				n.at<double>(j,2) = X[rand3.at(j)].z;
			}
			N = n.inv()*l;
		norm = sqrt(pow(N.at<double>(0,0), 2) + pow(N.at<double>(1,0), 2) + pow(N.at<double>(2,0), 2));
		N_norm = N/norm;						//The road norm-vector
		n2 = N_norm.at<double>(1,0);
		height = 1/norm;						//Resulting virtual height for the random sample
		if(n2 < -0.995)							//If similar enough to (0,-1,0), save
		{
			n1 = N_norm.at<double>(0,0);
			n3 = N_norm.at<double>(2,0);
			normal.push_back({n1,n2,n3});
			heights.push_back(height);
			u++;
		}
		}
		}
	//For analytical purposes i.e. the average generated height
	//could compared the most fit estimated by qScore
	double avgHeights = avg(heights);
	double varHeight = variance(heights, avgHeights);
	double sigHeight = sqrt(varHeight);
	if(heights.size() > 5)
	{
		int index;
		double h;
		tie(h,index) = qScore(heights);								//Returns the most fit virtual height and it's index
		acceptratio = heights.size()/(double)maxIter;	//How large share of the generated heights had acceptable road normals
		n1 = normal[index].x;
		n2 = normal[index].y;
		n3 = normal[index].z;
		Mat normVec = (Mat_<double>(1,3) << n1,n2,n3);
		return {h, normVec, heights, acceptratio, avgHeights, sigHeight};
	}
		return {avgHeights, normVec, heights, acceptratio, avgHeights, sigHeight};
	}

	vector<KeyPoint> shiTomasi(Mat img)
	{
		vector<KeyPoint> keypoints;
		std::vector<cv::Point2f> corners;
		Mat gray;
		cvtColor(img, gray, cv::COLOR_BGR2GRAY);
		int maxCorners = 1000;
		double qualityLevel = 0.05;
	  double minDistance = 3;
	  cv::Mat mask;
		int blockSize = 3;
		bool useHarrisDetector = false;
	  double k = 0.04;
		cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
		for(size_t i = 0; i < corners.size(); i++)
		{
		//cv::circle(img, corners[i], 3, cv::Scalar(255.), -1);
		}
		//cv::resize(img, img, cv::Size(), 1.5, 1.5);
	  //cv::imshow("Keypoints Shi-Tomasi", img);
		for(size_t i = 0; i < corners.size(); i++)
		{
	   keypoints.push_back(cv::KeyPoint(corners[i], 1.f));
		}
		return keypoints;
	}

	vector<KeyPoint> shiTomasiHelp(Mat img,
																 int maxCorners,
																 double qualityLevel,
																 int minDistance,
																 int blockSize,
																 bool useHarrisDetector,
															   double k)
	{
		vector<KeyPoint> keypoints;
		std::vector<cv::Point2f> corners;
		Mat gray;
		cvtColor(img, gray, cv::COLOR_BGR2GRAY);
	  cv::Mat mask;
		cv::goodFeaturesToTrack(gray, corners, maxCorners, qualityLevel, minDistance, mask, blockSize, useHarrisDetector, k);
		for(size_t i = 0; i < corners.size(); i++)
		{
		//cv::circle(img, corners[i], 3, cv::Scalar(255.), -1);
		}
		//cv::resize(img, img, cv::Size(), 1.5, 1.5);
	  //cv::imshow("Keypoints Shi-Tomasi", img);
		for(size_t i = 0; i < corners.size(); i++)
		{
	   keypoints.push_back(cv::KeyPoint(corners[i], 1.f));
		}
		return keypoints;
	}

	//Converts quaternions to euler angles
	tuple<float,float,float> Quat2Euler(float q0, float q1, float q2, float q3)
	{
		Mat RotVec;
		//Quaternion Rotation matrix R(q)
		Mat RotMat = (Mat_<double>(3,3) <<  q0*q0+q1*q1-q2*q2-q3*q3, 2*(q1*q2 - q0*q3), 2*(q0*q2 + q1*q3),
																		 		2*(q0*q3 + q1*q2), q0*q0-q1*q1+q2*q2-q3*q3, 2*(-q0*q1 + q2*q3),
																		 		2*(-q0*q2 + q1*q3), 2*(q2*q3 + q0*q1), q0*q0-q1*q1-q2*q2+q3*q3);
		//Simply convert to rotation vector and euler angles are found RotVec= [Roll,Pitch,Yaw]
		Rodrigues(RotMat, RotVec, noArray());
		return {RotVec.at<double>(0,0), RotVec.at<double>(1,0), RotVec.at<double>(2,0)}; //Return roll,pitch,yaw
	}

	//Generates random list of N = listsize unique elements in range 1-Max
	vector<int> randList(int listSize, int max)
	{
	    const int min = 1;        											//Min random number
	    vector<int> vec;
	    while(vec.size() != listSize)
		  {
				//vec.push_back(randnr(min, max));
				std::random_device r;
		    std::mt19937 gen(r());
		    std::uniform_int_distribution<> dis(min, max);
	      vec.emplace_back(dis(gen)); 									//Create new random number
	      std::sort(begin(vec), end(vec)); 							//Sort before call to unique
	      auto last = std::unique(begin(vec), end(vec));
	      vec.erase(last, end(vec));       							//Erase duplicates
	    }
	    //std::random_shuffle(begin(vec), end(vec)); 		//Mixes up the sequence
			return vec;
	}

//
////////// These functions below are not currently being used for the final algorithm
//

void testHomog(vector<double> hestimates, vector<Point2d> scene2, Mat cam, vector<Point3d> Xground)
{
	Mat Hom, Xprim, Err;
	double sad, hcur, e, hmin;
	double minerror = 100000000000;
	Mat Xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	for(uint16_t i = 0; i < hestimates.size(); i++)
	{
		hcur = hestimates[i];
		Hom = (Mat_<double>(3,3) << 1, 0, 0,
																0, 0, -hcur,
																0, 1, 0);
		sad = 0;
		e = 0;
		for(uint16_t k = 0; k < Xground.size(); k++)
		{
			Xk.at<double>(0,0) = Xground[k].x;
			Xk.at<double>(0,1) = Xground[k].z;
			xk.at<double>(0,0) = scene2[k].x;
			xk.at<double>(0,1) = scene2[k].y;
			Err = xk - cam*Hom*Xk;
			e = sqrt(pow(Err.at<double>(0,0),2) + pow(Err.at<double>(0,1),2) + pow(Err.at<double>(0,1),2));
			sad = sad + e;
		}
		if(e < minerror)
		{
			hmin = hcur;
		}
	}
	//cout << "Best estimate: " << 1.65/hmin << endl;
}

void orbDraw(Mat img)
{
	Ptr<ORB> ORBdraw = cv::ORB::create(1000,1.2f,8,5,0,2,ORB::FAST_SCORE,31,10);
	vector <KeyPoint> keyp;
	ORBdraw->detect(img, keyp, Mat());
	drawKeypoints(img, keyp, img);
	cv::resize(img, img, cv::Size(), 1.5, 1.5);
	cv::imshow("Keypoints ORB", img);
}

void PermGenerator(int n, int k)
{
    std::vector<int> d(n);
    std::iota(d.begin(),d.end(),1);
    //cout << "These are the Possible Permutations: " << endl;
    do
    {
        for (int i = 0; i < k; i++)
        {
            //cout << d[i] << " ";
        }
        //cout << endl;
        std::reverse(d.begin()+k,d.end());
    } while (next_permutation(d.begin(),d.end()));
}

void nelderMead(Point3d normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota)
{
	double n,n1,n2,n3,norm;
	Mat H;
	double nDiff = 0.2;
	double hDiff = 0.4;
	int iter = 100;
	double p = 2.5;
	double SAD;
	double min = 100000;
	double hbest;
	Mat x1 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat x2 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat err = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat N = cv::Mat::zeros(cv::Size(3,1), CV_64F);
	n1 = normalVec.x;
	n2 = normalVec.y;
	n3 = normalVec.z;
	N.at<double>(0,0) = n1;
	N.at<double>(0,1) = n2;
	N.at<double>(0,2) = n3;
	h = h - hDiff/2;
	for(uint16_t i = 0; i < iter; i++)
	{
		h = h + hDiff/iter;
		n = n3;
		n = n - nDiff/2;
		for(uint16_t j = 0; j < iter; j++)
		{
			n = n + nDiff/iter;
			N.at<double>(0,2) = n;
			norm = sqrt(pow(n1,2)+pow(n2,2)+pow(n,2));
			N = N/norm;
			H = cam*(rota + t*N/h)*cam.inv();
			SAD = 0;
			for(uint16_t k = 0; k < scene1.size(); k++)
			{
				x1.at<double>(0,0) = scene1[k].x;
				x1.at<double>(1,0) = scene1[k].y;
				x2.at<double>(0,0) = scene2[k].x;
				x2.at<double>(1,0) = scene2[k].y;
				err = x2 - H*x1;
				auto e = sqrt(pow(err.at<double>(0,0),2) + pow(err.at<double>(0,1),2) + pow(err.at<double>(0,2),2));
				SAD = SAD + e;
			}
			if(SAD < min)
			{
				hbest = h;
				min = SAD;
				//cout << "New best estimate: " << hbest << " i: " << i << " j: " << j << " SAD: " << SAD << endl;
			}
		}
	}
	//cout << "New best estimate: " << 1.65/hbest << endl;
}

void nelder(Mat normalVec, double h, Mat cam, vector<Point2d> scene1,vector<Point2d> scene2, Mat t, Mat rota)
{
	double n,n1,n2,n3,norm, theta, phi, htemp, m1, m2, m3;
	Mat H;
	double pi = 3.14159265359;
	double phDiff = 10;
	double thDiff = 10;
	double hDiff = 2;
	int iterh = 10;
	int iterth = 50;
	int iterphi = 50;
	double p = 25;
	double SAD;
	double min = 100000;
	double hbest;
	Mat x1 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat x2 = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat err = cv::Mat::ones(cv::Size(1,3), CV_64F);
	Mat N = cv::Mat::zeros(cv::Size(3,1), CV_64F);
	n1 = normalVec.at<double>(0,0);
	n2 = normalVec.at<double>(1,0);
	n3 = normalVec.at<double>(2,0);
	theta = acos(n3);
	phi = atan(n2/n1);

	theta = acos(n3) - thDiff*pi/360;
	for(uint16_t i = 0; i < iterth; i++)
	{
		theta = theta + thDiff*pi/360/iterth;
		phi = atan(n2/n1) - phDiff*pi/360;
		for(uint16_t j = 0; j < iterphi; j++)
		{
			phi = phi + phDiff*pi/360/iterphi;

			htemp = h - hDiff/2;
			for(uint16_t k = 0; k < iterh; k++)
			{
				htemp = htemp + hDiff/iterh;
				m1 = sin(theta)*cos(phi);
				m2 = sin(theta)*sin(phi);
				m3 = cos(theta);
				N.at<double>(0,0) = m1;
				N.at<double>(0,1) = m2;
				N.at<double>(0,2) = m3;
				H = cam*(rota + t*N/htemp)*cam.inv();
				SAD = 0;
				for(uint16_t v = 0; v < scene1.size(); v++)
				{
					x1.at<double>(0,0) = scene1[v].x;
					x1.at<double>(1,0) = scene1[v].y;
					x2.at<double>(0,0) = scene2[v].x;
					x2.at<double>(1,0) = scene2[v].y;
					err = x2 - H*x1;
					auto e = sqrt(pow(err.at<double>(0,0),2) + pow(err.at<double>(0,1),2) + pow(err.at<double>(0,1),2));
					SAD = SAD + e;
				}
				if(SAD < min)
				{
					hbest = htemp;
					min = SAD;
					//cout << k << endl;
					//cout << "New best estimate: " << hbest << " SAD: " << SAD << endl;
					//cout << "New best estimate: " << hbest << " i: " << i << " j: " << j << " SAD: " << SAD << endl;
				}
			}
		}
	}
}

void spherical(Point3d normvec)
{
double nx, ny, nz, theta, phi, phi2;
//cout << normvec << endl;
nx = normvec.x;
ny = normvec.y;
nz = normvec.z;
double pi = 3.14159265359;
theta = acos(nz);
phi = atan(ny/nx);
//cout << sin(theta)*cos(phi) << " " << sin(theta)*sin(phi) << " " << cos(theta) << endl;
}

Mat rot2Quat(Mat rotvec)
{
	double pitch,roll,yaw,q0,q1,q2,q3;
	pitch = rotvec.at<double>(0,0);
	roll = rotvec.at<double>(0,1);
	yaw = rotvec.at<double>(0,2);

	q0 = cos(pitch/2)*cos(roll/2)*cos(yaw/2) + sin(pitch/2)*sin(roll/2)*sin(yaw/2);
	q1 = sin(pitch/2)*cos(roll/2)*cos(yaw/2) - cos(pitch/2)*sin(roll/2)*sin(yaw/2);
	q2 = cos(pitch/2)*sin(roll/2)*cos(yaw/2) + sin(pitch/2)*cos(roll/2)*sin(yaw/2);
	q3 = cos(pitch/2)*cos(roll/2)*sin(yaw/2) - sin(pitch/2)*sin(roll/2)*cos(yaw/2);

	Mat q = (Mat_<double>(4,1) <<  q0,q1,q2,q3);
	return q;
}

void dispProjError(Mat Proj, vector<Point2d> pt2, vector<Point3d> p3d)
{
	Mat errorMat;
	double error;
	vector<double> errors;
	double averageError;
	Mat Xk = cv::Mat::ones(cv::Size(1,4), CV_64F);
	Mat xk = cv::Mat::ones(cv::Size(1,3), CV_64F);
	for(uint16_t i = 0; i < p3d.size(); i++)
	{
		Xk.at<double>(0,0) = p3d[i].x;
		Xk.at<double>(0,1) = p3d[i].y;
		Xk.at<double>(0,2) = p3d[i].z;
		// Form the current Image-point from input with index i
		xk.at<double>(0,0) = pt2[i].x;
		xk.at<double>(0,1) = pt2[i].y;

		errorMat = xk - Proj*Xk;
		error = sqrt(errorMat.at<double>(0,0)*errorMat.at<double>(0,0) + errorMat.at<double>(0,1)*errorMat.at<double>(0,1) + errorMat.at<double>(0,2)*errorMat.at<double>(0,2));
		if(i < 10000)
		{
		//cout << "Reprojection error: " << error << endl;
	  }
		errors.push_back(error);
	}
	averageError = avg(errors);
	//cout << "Average error: " << averageError << endl;
}

vector <vector <int>> combList(int n, int k)
{
		int len = binom(n, k);
		Mat c = cv::Mat::zeros(cv::Size(3,len), CV_8U);
		vector <vector <int>> list;
    std::string bitmask(k, 1); //k leading 1's
    bitmask.resize(n, 0); // n-k trailing 0's
    do {
				vector <int> temp;
        for (int i = 0; i < n; ++i) // [0..n-1] integers
        {
            if (bitmask[i])
						{
							temp.push_back(i);
						}
        }
				list.push_back(temp);
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
		return list;
}

void inliers(vector<Point3d> X, int order)
{
	int N = X.size();
	bool stat, statk;
	double aic,aicc,bic;
	vector<double>timevec, yVal;
	arma::gmm_diag model;
	arma::mat data(1, N, arma::fill::zeros);
	for(size_t i = 0; i <= N-1; i++){data(0,i) = X[i].y;}
	for(size_t k = 1; k <= order; k++)
	{
	//std::chrono::steady_clock::time_point startt = std::chrono::steady_clock::now();
	stat = model.learn(data, k, arma::eucl_dist, arma::random_subset, 10, 5, 1e-10, false);
	//std::chrono::steady_clock::time_point endt = std::chrono::steady_clock::now();
	//cout << std::chrono::duration_cast<std::chrono::microseconds>(endt - startt).count() << endl;
	//cout << endl;
	if(stat)
	{
		vector<double> means;
		for(uint16_t i = 0; i < k; i++){means.push_back(model.means(i));}
		auto it = minmax_element(begin(means), end(means));
		int minIdx = std::distance(means.begin(), it.first);
		//cout << "Order: " << k << " Lowest mean: " << means[minIdx] << " Weight: " << model.hefts(minIdx) << endl;
		sort(means.begin(), means.end());
		aic = 2*k - 2*model.sum_log_p(data);
		aicc = aic + (2*k*(k + 1))/(N - k - 1);
		bic = k*log(N) - 2*model.sum_log_p(data);
	}
}
}

//Converts to 4dim vector to to 3dim by dividing
//the first 3 parameters by the scale parameter
vector<Point3d> dim4to3(Mat dim4)
{
vector<Point3d> p3d;
for(uint16_t i = 0; i < dim4.cols; i++)
{
		double xs = dim4.at<double>(0,i)/dim4.at<double>(3,i);
		double ys = dim4.at<double>(1,i)/dim4.at<double>(3,i);
		double zs = dim4.at<double>(2,i)/dim4.at<double>(3,i);
		p3d.push_back({xs,ys,zs});
}
return p3d;
}

Mat normalizeVec(Mat tvec)
{
	double normal;
	normal = pow(tvec.at<double>(0,0),2) + pow(tvec.at<double>(1,0),2) + pow(tvec.at<double>(2,0),2);
	return tvec/sqrt(normal);
}

int ran(int min, int max)
{
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

int randnr(int min, int max)
{
    std::random_device r;
    std::mt19937 gen(r());
    std::uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

vector<int> kthCombination(int n, int k, int m)
{
	vector<int> result;
	int a = n; //4
	int b = k; //2
	int x = (binom(n, k) - 1) - m; //3
	for(uint16_t i = 0; i < k; i++)
	{
		a--;
		while(binom(a,b) > x)
		{
			a--;
		}
			result.push_back(n - 1 - a);
			//cout << n - a << ",";
			x = x - binom(a, b);
			b--;
	}
	 //cout << endl;
	 return result;
}

vector<double> removeLoners(vector<double> val, double dist)
{
	vector<size_t> idx(val.size());
	iota(idx.begin(), idx.end(), 0);
	stable_sort(idx.begin(), idx.end(), [&val](size_t i1, size_t i2) {return val[i1] < val[i2];});
	double n;
	vector<double> near;
	for(uint16_t i = 0; i < val.size()-1; i++)
	{
		n = val[idx[i+1]]-val[idx[i]];
		near.push_back(n);
	}
	double mean = avg(near);
	vector<double> filt;
	for(uint16_t i = 0; i < val.size()-1; i++)
		{
			switch(i)
			{
  		case 0:
			if(near[i] < dist*mean)
			{
				filt.push_back(val[idx[i]]);
			}
    	break;
  		default:
			if(min(near[i-1],near[i]) < dist*mean)
			{
				filt.push_back(val[idx[i]]);
			}
			}
		}
		return filt;
}
/*
status = model.learn(data, 1, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
if(status)
{
	k = 1;
	aic1 = 2*k - 2*model.sum_log_p(data);
	aicc1 = aic1 + (2*k*(k + 1))/(N - k - 1);
	bic1 = k*log(N) - 2*model.sum_log_p(data);
	if(bic1 < minScore)
	{
		minScore = bic1;
	}
}
status2 = model.learn(data, 2, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
if(status2)
{
k = 2;
double mean0 = model.means(0);
double mean1 = model.means(1);
vector<double> means;
means.push_back(mean0);
means.push_back(mean1);
auto it = minmax_element(begin(means), end(means));
int minIdx = std::distance(means.begin(), it.first);
minw = model.hefts(minIdx);
aic2 = 2*k - 2*model.sum_log_p(data);
aicc2 = aic2 + (2*k*(k + 1))/(N - k - 1);
bic2 = k*log(N) - 2*model.sum_log_p(data);
if(bic2 < minScore && minw > minimumweight)
{
	minScore = bic2;
	order = 2;
	mu = model.means[minIdx];
	sigma = sqrt(model.dcovs[minIdx]);
}
}
status3 = model.learn(data, 3, arma::maha_dist, arma::random_subset, 10, 5, 1e-10, false);
if(status3)
{
k = 3;
double mean0 = model.means(0);
double mean1 = model.means(1);
double mean2 = model.means(2);
vector<double> means;
means.push_back(mean0);
means.push_back(mean1);
means.push_back(mean2);
auto it = minmax_element(begin(means), end(means));
int minIdx = std::distance(means.begin(), it.first);
sort(means.begin(), means.end());
minw = model.hefts(minIdx);
mean0 = means[0];
mean1 = means[1];
mean2 = means[2];
aic3 = 2*k - 2*model.sum_log_p(data);
aicc3 = aic3 + (2*k*(k + 1))/(N - k - 1);
bic3 = k*log(N) - 2*model.sum_log_p(data);
if(bic3 < minScore && minw > minimumweight)
{
order = 3;
mu = model.means[minIdx];
sigma = sqrt(model.dcovs[minIdx]);
}
}*/
