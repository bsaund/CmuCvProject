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


bool hasDepthSpike(const Mat &disp, int start, int width){
	float maxD = 0, leftD = 0, rightD = 0;
	int end = start+width;
	int y = start/disp.cols;
	int lx = start % disp.cols;
	int rx = lx + width;
	int i = 0;

	while(leftD == 0){
		i++;
		if((lx-i) < 0)
			return false;
		leftD = disp.at<float>(start-i);
	}
	i=0;
	while(rightD == 0){
		i++;
		if(rx+i > disp.cols)
			return false;
		rightD = disp.at<float>(end + i);
	}


	for(int ind = start; ind<end; ind++){
		float d = disp.at<float>(ind);
		if(d != 0){
			maxD = max(d, maxD);
		}
	}
	if(maxD == 0)
		return false;

	float spike = max(leftD, rightD)/maxD;
	
	return spike < 0.5;
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
	scale = max(min(scale, 1.3), .7);

	for(int i = 1; i < rightInd-leftInd; i++){
		img.at<Vec3b>(y,rightInd - i) = rightImg.at<Vec3b>(rImgR - i) * scale;
		// img.at<Vec3b>(y,rightInd - i) = rightImg.at<Vec3b>(rImgR - i);
	}

}

void fillInGreen(Mat &img, int y, int leftInd, int rightInd){

	for(int i = 1; i < rightInd-leftInd; i++){
		img.at<Vec3b>(y,rightInd - i) = Vec3b(0, 255, 0);
	}

}

void fillInMissingCorr(Mat &img, Mat &depthImg, const Mat &leftImg, Mat &rightImg,
											 Mat &lFrom, Mat &rFrom, Mat &dispLTmp, Mat &dispRTmp){
	int numFill = 0;
	Mat dispL = dispLTmp.clone();
	Mat dispR = dispRTmp.clone();

	dispLTmp = Mat();
	dispRTmp = Mat();
	cvtColor(dispL, dispLTmp, CV_GRAY2RGB);
	cvtColor(dispR, dispRTmp, CV_GRAY2RGB);

	
	for(int y=1; y<img.rows-1; y++){
		float rd = 0;
		float ld = 0;
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
			fillInGreen(dispLTmp, y, leftInd, rightInd);
			// if(abs((rd-ld)/rd) < .1){
			// 	fillInL(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
			// 	continue;
			// }
			

			int llind = lFrom.at<int>(y, leftInd);
			int lrind = rFrom.at<int>(y, leftInd);
			int width = rightInd - leftInd;

			int intense = 100;

			// fillInL(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
			// continue;
			if(hasDepthSpike(dispR, lrind, width)){
				fillInL(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
				// fillInGreen(img, y, leftInd, rightInd);
				for(int i=0; i<width; i++){
					dispLTmp.at<Vec3f>(lrind + i) = Vec3f(intense, 0, 0);
				}
				// fillInGreen(dispLTmp, y, leftInd, rightInd);

				continue;
			}


			// continue;

			if(hasDepthSpike(dispL, llind, width)){
				numFill++;
				fillInR(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
				// fillInGreen(img, y, leftInd, rightInd);
				for(int i=0; i<width; i++){
					dispLTmp.at<Vec3f>(llind + i) = Vec3f(0, intense, 0);
				}

				continue;
			}
			// continue;


			if(ld < rd){
				fillInR(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
				for(int i=0; i<width; i++){
					dispLTmp.at<Vec3f>(llind + i) = Vec3f(0, intense, 0);
				}

				continue;
			}
			fillInL(img, leftImg, rightImg, lFrom, rFrom, y, leftInd, rightInd);
			for(int i=0; i<width; i++){
				dispLTmp.at<Vec3f>(llind + i) = Vec3f(intense, 0, 0);
			}


		}
	}	
	namedWindow("depth", 1);
	imshow("depth", dispLTmp);
	imwrite("disp_filled.jpg", dispLTmp);

	printf("Num fill: %d\n", numFill);
}

void postFillIn(Mat &img, Mat &depthImg, Mat &leftImg, Mat &rightImg,
								Mat &lFrom, Mat &rFrom, Mat &dispL, Mat &dispR){

	fillInMissingCorr(img, depthImg, leftImg, rightImg, lFrom, rFrom, dispL, dispR);
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
														 Mat &dispL, Mat &dispR,
														 Mat &newImage, Mat &depthImage, Mat &combImg) {


	Mat _3dImage, _3dImageTmp;


	reprojectImageTo3D(dispL, _3dImageTmp, p.Q, true);
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
	Mat depthImageR = Mat::zeros(img1_colored.size(), CV_32F);
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
		depthImageR.at<float>(y, x) = _3dImage.at<Vec3f>(0,i-(int)dispL.at<float>(i))[2];
		cameFromL.at<int>(y,x) = i;
		cameFromR.at<int>(y,x) = i - (int)dispL.at<float>(i);
		

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

	Mat dispLC = dispL.clone();
	Mat dispRC = dispR.clone();


	postFillIn(newImage, depthImage, img1_colored, img2_colored, 
						 cameFromL, cameFromR, dispLC, dispRC);

	// namedWindow("postFill", 1);
	// imshow("postFill", newImage);

	Size sz = img1_colored.size();
	combImg = Mat(sz.height*2, sz.width*2, CV_8UC3);
	Mat ul(combImg, Rect(0,0,sz.width,sz.height));
	Mat ur(combImg, Rect(sz.width,0,sz.width,sz.height));
	Mat ll(combImg, Rect(0,sz.height,sz.width,sz.height));
	Mat lr(combImg, Rect(sz.width,sz.height,sz.width,sz.height));
	img1_colored.copyTo(ul);
	// img2_colored.copyTo(ur);
	Mat cDisp;
	cvtColor(depthImage, cDisp, CV_GRAY2RGB);
	cDisp.convertTo(ur, CV_8UC3);
	repro.convertTo(ll, CV_8UC3);
	newImage.convertTo(lr, CV_8UC3);

	rectangle(ul, cvPoint(20,40), cvPoint(160,10), cvScalar(0,0,0), CV_FILLED);
	putText(ul, "Left Camera", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
					cvScalar(250,250,250), 1, CV_AA);
	putText(ur, "Depth", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
					cvScalar(250,250,250), 1, CV_AA);
	putText(ll, "3D", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
					cvScalar(250,250,250), 1, CV_AA);
	rectangle(lr, cvPoint(0,0), cvPoint(176, 480), cvScalar(0,0,0), CV_FILLED);
	putText(lr, "Filled", cvPoint(30,30), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
					cvScalar(250,250,250), 1, CV_AA);
	putText(lr, "Virtual", cvPoint(30,60), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
					cvScalar(250,250,250), 1, CV_AA);
	putText(lr, "Panning", cvPoint(30,90), FONT_HERSHEY_COMPLEX_SMALL, 0.8, 
					cvScalar(250,250,250), 1, CV_AA);

	imwrite("unfilled.jpg", ll);
	imwrite("filled.jpg", lr);

	// Mat tmp;
	// newImage.convertTo(tmp, CV_8UC3);
	// GaussianBlur(tmp, lr, Size(5,5), 0, 5);



}


