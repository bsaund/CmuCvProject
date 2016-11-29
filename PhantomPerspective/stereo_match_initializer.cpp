#include "stereo_matcher_initializer.h"



bool loadCameraParams(std::string intrinsic_filename, std::string extrinsic_filename,
	trinsics &p) {
	if (intrinsic_filename.empty() || extrinsic_filename.empty()) {
		return false;
	}
	// reading intrinsic parameters
	cv::FileStorage fs(intrinsic_filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		printf("Failed to open file %s\n", intrinsic_filename.c_str());
		return false;
	}

	fs["M1"] >> p.M1;
	fs["D1"] >> p.D1;
	fs["M2"] >> p.M2;
	fs["D2"] >> p.D2;

	int scale = 1;
	p.M1 *= scale;
	p.M2 *= scale;

	fs.open(extrinsic_filename, cv::FileStorage::READ);
	if (!fs.isOpened()) {
		printf("Failed to open file %s\n", extrinsic_filename.c_str());
		return false;
	}

	fs["R"] >> p.R;
	fs["T"] >> p.T;
	return true;

};

cv::Ptr<cv::StereoBM> getStereoBM(cv::Rect roi1, cv::Rect roi2, int SADWindowSize,
	int numberOfDisparities) {
	cv::Ptr<cv::StereoBM> bm = cv::StereoBM::create(16, 9);
	bm->setROI1(roi1);
	bm->setROI2(roi2);
	bm->setPreFilterCap(31);
	bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
	bm->setMinDisparity(0);
	bm->setNumDisparities(numberOfDisparities);
	bm->setTextureThreshold(10);
	bm->setUniquenessRatio(15);
	bm->setSpeckleWindowSize(100);
	bm->setSpeckleRange(32);
	bm->setDisp12MaxDiff(1);
	return bm;
};

cv::Ptr<cv::StereoSGBM> getStereoSGBM(int SADWindowSize, int numberOfDisparities, int cn, int alg) {
	int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;

	cv::Ptr<cv::StereoSGBM> sgbm = cv::StereoSGBM::create(0, 16, 3);
	sgbm->setPreFilterCap(63);
	sgbm->setBlockSize(sgbmWinSize);
	sgbm->setP1(8 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setP2(32 * cn*sgbmWinSize*sgbmWinSize);
	sgbm->setMinDisparity(0);
	sgbm->setNumDisparities(numberOfDisparities);
	sgbm->setUniquenessRatio(10);
	sgbm->setSpeckleWindowSize(100);
	sgbm->setSpeckleRange(32);
	sgbm->setDisp12MaxDiff(1);

	if (alg == STEREO_HH)
		sgbm->setMode(cv::StereoSGBM::MODE_HH);
	else if (alg == STEREO_SGBM)
		sgbm->setMode(cv::StereoSGBM::MODE_SGBM);
	else if (alg == STEREO_3WAY)
		sgbm->setMode(cv::StereoSGBM::MODE_SGBM_3WAY);


	return sgbm;
};