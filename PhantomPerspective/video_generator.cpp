/*
 *	stereo_match_single.cpp
 *
 *
 */

//PhantomPerspective.exe 1 2 -i=../../intrinsics.yml -e=../../extrinsics.yml --blocksize=9 --max-disparity=320 --algorithm=bm

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/core/utility.hpp"
#include "stereo_matcher_initializer.h"
#include "perspective_matcher.h"
#include "opencv2/ximgproc/disparity_filter.hpp"

#include <stdio.h>

using namespace cv;
using namespace cv::ximgproc;

int singleDepthMap(Mat img1, Mat img2, Mat img1_colored, Mat img2_colored, 
									 Ptr<StereoMatcher> sgbm, int alg,
									 maps m, trinsics &p, VideoWriter &vid);


int runWriter(std::string img1_filename, std::string img2_filename,
							VideoWriter &vid){

	std::string intrinsic_filename = "testImgs/intrinsics.yml";
	std::string extrinsic_filename = "testImgs/extrinsics.yml";



	int SADWindowSize, numberOfDisparities;

	// int alg = STEREO_BM;
	int alg =STEREO_SGBM;
	// STEREO_HH;
	// STEREO_VAR;
	// STEREO_3WAY;

	numberOfDisparities = 176;
	// numberOfDisparities = 128;
	// numberOfDisparities = 256;
	SADWindowSize = 3;

	int color_mode = alg == STEREO_BM ? 0 : -1;

	trinsics p;
	if (!loadCameraParams(intrinsic_filename, extrinsic_filename, p)) {
		printf("Problem loading intrinsics or extrinsics\n");
		return -1;
	}


  Mat img1_colored = imread(img1_filename, CV_LOAD_IMAGE_COLOR);
  Mat img2_colored = imread(img2_filename, CV_LOAD_IMAGE_COLOR);

  Mat img1 = imread(img1_filename, color_mode);
  Mat img2 = imread(img2_filename, color_mode);

	Rect roi1, roi2;
	Size img_size = img1.size();
	stereoRectify(p.M1, p.D1, p.M2, p.D2, img_size, p.R, p.T, p.R1, p.R2, p.P1, p.P2, p.Q,
								CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

	maps m;
	initUndistortRectifyMap(p.M1, p.D1, p.R1, p.P1, img_size, CV_16SC2, m.map11, m.map12);
	initUndistortRectifyMap(p.M2, p.D2, p.R2, p.P2, img_size, CV_16SC2, m.map21, m.map22);


	int cn = img1.channels();
	Ptr<StereoMatcher> usedBm;

	if (alg == STEREO_BM)
		usedBm = getStereoBM(roi1, roi2, SADWindowSize, numberOfDisparities);
	else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
		usedBm = getStereoSGBM(SADWindowSize, numberOfDisparities, cn, alg);



	singleDepthMap(img1, img2, img1_colored, img2_colored, usedBm, alg, m, p, vid);
}

int singleDepthMap(Mat img1, Mat img2, Mat img1_colored, Mat img2_colored, 
									 Ptr<StereoMatcher> sgbm, int alg,
									 maps m, trinsics &p, VideoWriter &vid) {
	namedWindow("left", 1);
	namedWindow("right", 1);
	namedWindow("disparity", 1);
	namedWindow("newPerspective", 1);
	namedWindow("newDepth", 1);
	namedWindow("final", 1);

	Ptr<DisparityWLSFilter> wls_filter;


	Mat orig1 = img1.clone();
	Mat orig2 = img2.clone();

	
	Mat dispL, dispR, newImage, depth_img;
 
	Mat R = Mat::eye(3, 3, cv::DataType<double>::type);
	Mat T = Mat::zeros(3, 1, cv::DataType<double>::type);

	double movement = 0;
	double increment = 0.011;

	char charCheckForEsc = 0;

	img1 = orig1.clone();
	img2 = orig2.clone();
	rectifyBoth(img1, img2, m);
	rectifyBoth(img1_colored, img2_colored, m);

	Mat dispInt, dispIntFilt;
	sgbm->compute(img1, img2, dispInt);
	dispInt.convertTo(dispL, CV_32F);
	dispL /= 16;  //sgbm returns disp as a 4-fractional-bit short


	
	// Ptr<StereoMatcher> right_matcher = createRightMatcher(sgbm);
	// wls_filter = createDisparityWLSFilter(sgbm);
	// right_matcher->compute(img2, img1, dispInt);
	// dispInt.convertTo(dispR, CV_32F);
	// dispR /= 16;
	dispR = Mat::zeros(dispL.size(), CV_32F);

	for(int y = 0; y<dispL.rows; y++){
		for(int x = 0; x<dispL.cols; x++){
			float disparity = dispL.at<float>(y,x);
			if(disparity == 0)
				continue;
			dispR.at<float>(y,x-disparity) = disparity;
		}
	}


	for(int i=0; i<175; i++){
		movement += increment;
		if(movement >= 1 || movement <= 0){
			increment *= -1;
		}

		
		Mat T = movement*p.T;
		Mat display;
		getDifferentPerspective(img1_colored, img2_colored,
														R, T, p, dispL, dispR,
														newImage, depth_img, display);

		int numDisp = sgbm->getNumDisparities();

		imshow("final", display);
		vid << display;
		
		charCheckForEsc = cv::waitKey(1);		// delay (in ms) and get key press, if any
		if(charCheckForEsc == 27)
			return 1;
	}

	return 1;

}


int main(int argc, char** argv)
{
	std::string img1_filename = "testImgs/2_left.jpg";
	std::string img2_filename = "testImgs/2_right.jpg";

	VideoWriter vid;
  Mat img1_colored = imread(img1_filename, CV_LOAD_IMAGE_COLOR);
	Size sz = img1_colored.size();
	Size fullSz = Size(sz.width*2, sz.height*2);
	int fourcc = CV_FOURCC('H', '2', '6', '4');
	vid.open("final.mp4", fourcc, 24, fullSz);
	
	for(int i=1; i<=8; i++){
		std::ostringstream ossL, ossR;
		ossL << "testImgs/" << i << "_left.jpg";
		ossR << "testImgs/" << i << "_right.jpg";
		runWriter(ossL.str(), ossR.str(), vid);
	}
	return 0;
}
