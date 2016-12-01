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



void postFillIn(Mat &img){
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


void getDifferentPerspective(Mat img1_colored, Mat img2_colored, 
														 Mat &R, Mat &T,
														 trinsics &p, 
														 Mat &disp, Mat &newImage) {

	Mat _3dImage;
	reprojectImageTo3D(disp, _3dImage, p.Q, true);
	Mat imagePoints;

	//cout << img1_colored.type();
	_3dImage = _3dImage.reshape(3, 1);
	// projectPoints(_3dImage, Mat::eye(3, 3, cv::DataType<double>::type),
	// 	Mat::zeros(3, 1, cv::DataType<double>::type), p.M1, Mat(),
	// 	imagePoints);
	projectPoints(_3dImage, R, T, p.M1, Mat(), imagePoints);
 		
	//printf("rows: %d, cols: %d, dims: %d, ", _3dImage.rows, _3dImage.cols, _3dImage.dims);
	//printf("channels: %d\n", _3dImage.channels());
	//printf("rows: %d, cols: %d, dims: %d, chan: %d", imagePoints.rows, imagePoints.cols, imagePoints.dims, imagePoints.channels());

	newImage = Mat::zeros(img1_colored.size(), img1_colored.type());
	for (int i = 0; i < imagePoints.rows; i++) {				
		//printf("depth: %f\n", _3dImage.at<Vec3f>(0,i)[2]);
		if (_3dImage.at<Vec3f>(0, i)[2] > 100)
			continue;
	
		Vec2f v = imagePoints.at<Vec2f>(i, 0);
		int x = round(v[0]);
		int y = round(v[1]);
		
		if (x < 0 || y < 0 || y >= newImage.rows || x >= newImage.cols)
			continue;
		newImage.at<Vec3b>(y, x) =  img1_colored.at<Vec3b>(i);		
	}
	/*for (int row = 0; row < newImage.rows; row++){
		for (int col = 0; col < newImage.cols; col++) {

		}
	}*/
	postFillIn(newImage);
}


