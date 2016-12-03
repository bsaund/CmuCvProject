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


/*
 *  Returns true if there is any non-zero element in element 
 *  of matrix img (y,x)
 */
bool isFilled(const Mat &img, int y, int x){
	const uchar* val = img.ptr(y,x);
	for(int i=0; i<img.elemSize(); i++){
		if(val[i])
			return true;
	}
	return false;
}

/*
 *  Fill in single pixel gaps by averaging surrounding pixels
 */
template<typename T>
void fillInLines(Mat &img, bool average=true){
	for(int y=1; y<img.rows-1; y++){
		for(int x=1; x<img.cols-1; x++){
			if(isFilled(img, y, x))
				continue;

			//fill in horizontal lines from aliasing
			if(isFilled(img, y-1, x) && isFilled(img, y+1, x)){
				if(average)
					img.at<T>(y,x) = img.at<T>(y-1, x)/2 + img.at<T>(y+1, x)/2;
				else
					img.at<T>(y,x) = img.at<T>(y-1, x);
				continue;
			}

			//fill in veritcal lines from aliasing
			if(isFilled(img, y, x-1) && isFilled(img, y, x+1)){
				if(average)
					img.at<T>(y,x) = img.at<T>(y, x-1)/2 + img.at<T>(y, x+1)/2;
				else
					img.at<T>(y,x) = img.at<T>(y, x-1);
				continue;
			}
		}
	}
}


void fillInL(Mat &img, const Mat &leftImg, Mat &rightImg, Mat &lFrom, Mat &rFrom,
						 int y, int leftInd, int rightInd){
int lImgL = lFrom.at<int>(y,leftInd);
 for(int i = 1; i < rightInd-leftInd; i++){
	 img.at<Vec3b>(y,leftInd + i) = leftImg.at<Vec3b>(lImgL + i);
 }
}

void fillInR(Mat &img, const Mat &leftImg, Mat &rightImg, Mat &lFrom, Mat &rFrom,
						 int y, int leftInd, int rightInd){
	int rImgR = rFrom.at<int>(y,rightInd);
	Vec3b lpix = leftImg.at<Vec3b>(lFrom.at<int>(y,rightInd));
	Vec3b rpix = rightImg.at<Vec3b>(rImgR);
	double scale = norm(lpix, CV_L2)/ norm(rpix, CV_L2);

	for(int i = 1; i < rightInd-leftInd; i++){
		img.at<Vec3b>(y,rightInd - i) = rightImg.at<Vec3b>(rImgR - i) * scale;
	}

}

void fillInMissingCorr(Mat &img, Mat &depthImg, const Mat &leftImg, Mat &rightImg,
											 Mat &lFrom, Mat &rFrom){
	int numFill = 0;
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

			fillingPatch = false;			
			

			

			rd = depthImg.at<float>(y,x);
			int rightInd = x;
			

			// if(rightInd - leftInd > 5)
			// 	continue;

			// Mat rightFill = Mat::zeros(img1_colored.size(), img1_colored.type());;
			// printf("width %d\n", rightInd - leftInd);
			if(leftInd == 0)
				continue;

			

			if(ld < rd){
				fillInR(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
				continue;
			}
			fillInL(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);

		}
	}	
	printf("Num fill: %d\n", numFill);
}

void postFillIn(Mat &img, Mat &depthImg, Mat &leftImg, Mat &rightImg,
								Mat &lFrom, Mat &rFrom){

	fillInMissingCorr(img, depthImg, leftImg, rightImg, lFrom, rFrom);
}

void preFilterNewImg(Mat &img){
	for(int x=0; x<img.cols; x++){
		bool topFilled = true;
		int stretch = 0;
		for(int y=1; y<img.rows-1; y++){
			// if(!topFilled && img.at<float>(y+1,x) == 0){
			// 	img.at<float>(y,x) = 0;
			// }
			if(stretch < 5 && !isFilled(img, y+1,x)){
				for(;stretch>=0;stretch--){
					img.at<Vec3b>(y-stretch,x) = Vec3b(0,0,0);
				}
			}
			if(!isFilled(img, y,x)){
				stretch++;
			} else {
				stretch = 0;
			}
			// topFilled = isFilled(img,
			// topFilled = (img.at<float>(y,x) != 0);
		}
	}
}


void preFilterDisp(Mat &disp){
	for(int x=0; x<disp.cols; x++){
		bool topFilled = true;
		int stretch = 0;
		for(int y=1; y<disp.rows-1; y++){
			// if(!topFilled && disp.at<float>(y+1,x) == 0){
			// 	disp.at<float>(y,x) = 0;
			// }
			if(stretch < 5 && disp.at<float>(y+1,x)==0){
				for(;stretch>=0;stretch--){
					disp.at<float>(y-stretch,x) = 0;
				}
			}
			if(disp.at<float>(y,x) == 0){
				stretch++;
			} else {
				stretch = 0;
			}
			// topFilled = isFilled(img,
			// topFilled = (disp.at<float>(y,x) != 0);
		}
	}
}


void getDifferentPerspective(Mat img1_colored, Mat img2_colored, 
							 Mat &R, Mat &T,
							 trinsics &p, 
														 Mat &disp, Mat &newImage, Mat &depthImage, Mat &combImg) {


	Mat _3dImage, _3dImageTmp;


	reprojectImageTo3D(disp, _3dImageTmp, p.Q, true);
	Mat imagePointsTmp;
	Mat cameFromL, cameFromR;

	_3dImageTmp = _3dImageTmp.reshape(3, 1);
	Mat sortedInd;
	Mat chan[3];
	split(_3dImageTmp, chan);
	Mat _3dSorted = Mat::zeros(_3dImageTmp.size(), _3dImageTmp.type());;

	sortIdx(chan[2], sortedInd, CV_SORT_EVERY_ROW + CV_SORT_DESCENDING);	
	for(int i=0; i<_3dImageTmp.cols; i++){
		_3dSorted.at<Vec3f>(i) = _3dImageTmp.at<Vec3f>(sortedInd.at<int>(i));
		// printf("%f\n", chan[2].at<float>(sortedInd.at<int>(i)));
		// printf("%d\n", sortedInd.at<int>(i));
	}
	// _3dSorted.at<Vec3f>(0) = _3dImage.at<Vec3f>(0);

	_3dImage = _3dImageTmp;
	projectPoints(_3dSorted, R, T, p.M1, Mat(), imagePointsTmp);
	Mat imagePoints = imagePointsTmp.clone();
	for(int i=0; i<_3dImageTmp.cols; i++){
		imagePoints.at<Point2f>(sortedInd.at<int>(i)) = imagePointsTmp.at<Point2f>(i);
	}
		
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
	// namedWindow("preLine", 1);
	// imshow("preLines", newImage);

	Mat repro = newImage.clone();

	fillInLines<Vec3b>(newImage, false);
	fillInLines<float>(depthImage, false);
	fillInLines<int>(cameFromL, false);
	fillInLines<int>(cameFromR, false);


	// namedWindow("postLines", 1);
	// imshow("postLines", newImage);


	preFilterDisp(depthImage);
	preFilterNewImg(newImage);

	// namedWindow("postFilt", 1);
	// imshow("postFilt", newImage);

	postFillIn(newImage, depthImage, img1_colored, img2_colored, 
						 cameFromL, cameFromR);

	// namedWindow("postFill", 1);
	// imshow("postFill", newImage);

	Size sz = img1_colored.size();
	combImg = Mat(sz.height*2, sz.width*2, CV_8UC3);
	Mat ul(combImg, Rect(0,0,sz.width,sz.height));
	Mat ur(combImg, Rect(sz.width,0,sz.width,sz.height));
	Mat ll(combImg, Rect(0,sz.height,sz.width,sz.height));
	Mat lr(combImg, Rect(sz.width,sz.height,sz.width,sz.height));
	img1_colored.copyTo(ul);
	img2_colored.copyTo(ur);
	repro.convertTo(ll, CV_8UC3);
	newImage.convertTo(lr, CV_8UC3);



}


