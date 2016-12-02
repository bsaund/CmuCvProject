#include "perspective_matcher.h"
#include<iostream>
using namespace std;
using namespace cv;

void rectifyBoth(Mat &img1, Mat &img2, const maps &m){
	Mat img1r, img2r;
	remap(img1, img1r, m.map11, m.map12, INTER_LINEAR);
	remap(img2, img2r, m.map21, m.map22, INTER_LINEAR);
	img1 = img1r;
	img2 = img2r;
}



bool isFilled(const Mat &img, int y, int x){
	Vec3b p = img.at<Vec3b>(y, x);
	return p[0] || p[1] || p[2];
}


void fillInLines(Mat &img){
	for(int y=1; y<img.rows-1; y++){
		for(int x=1; x<img.cols-1; x++){
			if(isFilled(img, y, x))
				continue;

			//fill in horizontal lines from aliasing
			if(isFilled(img, y-1, x) && isFilled(img, y+1, x)){
				img.at<Vec3b>(y,x) = img.at<Vec3b>(y-1, x)/2 + img.at<Vec3b>(y+1, x)/2;
				continue;
			}


			//fill in veritcal lines from aliasing
			if(isFilled(img, y, x-1) && isFilled(img, y, x+1)){
				img.at<Vec3b>(y,x) = img.at<Vec3b>(y, x-1)/2 + img.at<Vec3b>(y, x+1)/2;
				continue;
			}
		}
	}
}

void fillInMissingCorr(Mat &img, Mat &depthImg, Mat &leftImg, Mat &rightImg,
											 Mat &lFrom, Mat &rFrom){
	for(int y=1; y<img.rows-1; y++){
		int rd = 0;
		int ld = 0;
		bool fillingPatch = false;
		int leftInd = 0;
		for(int x=1; x<img.cols-1; x++){
			if(!isFilled(img, y, x)){
				if(!fillingPatch){
					leftInd = x-1;
					ld = depthImg.at<float>(y,x-1);
				}
				fillingPatch = true;
				continue;
			}
			
			if(!fillingPatch)
				continue;

			rd = depthImg.at<float>(y,x);
			int rightInd = x;

			if(ld < rd){
				if(leftInd == 0){
					int lImgL = lFrom.at<int>(y,rightInd);
					for(int i = 1; i < rightInd-leftInd; i++){
						img.at<Vec3b>(y,rightInd - i) = leftImg.at<Vec3b>(lImgL - i);
						// img.at<Vec3b>(y,rightInd - i) = leftImg.at<Vec3b>(y, rightInd- i);
					}
				} else {
					int rImgR = rFrom.at<int>(y,rightInd);
					for(int i = 1; i < rightInd-leftInd; i++){
						// for(int ind = leftInd+1; ind < rightInd; ind++){
						img.at<Vec3b>(y,rightInd - i) = rightImg.at<Vec3b>(rImgR - i);
					}

				}
			} else {
				int lImgL = lFrom.at<int>(y,leftInd);
				for(int i = 1; i < rightInd-leftInd; i++){
					img.at<Vec3b>(y,leftInd + i) = leftImg.at<Vec3b>(lImgL + i);
				}
			}


			rd = 0;
			ld = 0;
			fillingPatch = false;
		}
	}	
}

void postFillIn(Mat &img, Mat &depthImg, Mat &leftImg, Mat &rightImg,
								Mat &lFrom, Mat &rFrom){
	fillInLines(img);
	fillInMissingCorr(img, depthImg, leftImg, rightImg, lFrom, rFrom);
}


void getDifferentPerspective(Mat img1_colored, Mat img2_colored, 
							 Mat &R, Mat &T,
							 trinsics &p, 
							 Mat &disp, Mat &newImage, Mat &depthImage) {


	Mat _3dImage;
	reprojectImageTo3D(disp, _3dImage, p.Q, true);
	Mat imagePoints;
	Mat cameFromL, cameFromR;

	_3dImage = _3dImage.reshape(3, 1);

	projectPoints(_3dImage, R, T, p.M1, Mat(), imagePoints);
		
	newImage = Mat::zeros(img1_colored.size(), img1_colored.type());
	depthImage = Mat::zeros(img1_colored.size(), CV_32F);
	cameFromL = Mat::zeros(img1_colored.size(), CV_32S);
	cameFromR = Mat::zeros(img1_colored.size(), CV_32S);
	for (int i = 0; i < imagePoints.rows; i++) {				
		//printf("depth: %f\n", _3dImage.at<Vec3f>(0,i)[2]);
		if (_3dImage.at<Vec3f>(0, i)[2] > 1000)
			continue;
	
		Vec2f v = imagePoints.at<Vec2f>(i, 0);
		int x = round(v[0]);
		int y = round(v[1]);
		
		if (x < 0 || y < 0 || y >= newImage.rows || x >= newImage.cols)
			continue;

		//These two lines do almost the same thing, since the colors should match for the two images
		newImage.at<Vec3b>(y, x) = img1_colored.at<Vec3b>(i);
		// newImage.at<Vec3b>(y, x) = img2_colored.at<Vec3b>(i - disp.at<float>(i));
		
		depthImage.at<float>(y, x) = _3dImage.at<Vec3f>(0,i)[2];
		cameFromL.at<int>(y,x) = i;
		cameFromR.at<int>(y,x) = i - (int)disp.at<float>(i);
		

	}
	postFillIn(newImage, depthImage, img1_colored, img2_colored, 
						 cameFromL, cameFromR);
	
}


