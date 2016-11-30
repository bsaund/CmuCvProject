/*
 *  stereo_match_single.cpp
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

#include <stdio.h>

using namespace cv;




static void print_help()
{
  printf("\nDemo stereo matching converting L and R images into disparity and point clouds\n");
  printf("\nUsage: stereo_match <left_image> <right_image> [--algorithm=bm|sgbm|hh|sgbm3way] [--blocksize=<block_size>]\n"
	 "[--max-disparity=<max_disparity>] [--scale=scale_factor>] [-i=<intrinsic_filename>] [-e=<extrinsic_filename>]\n"
	 "[--no-display] [-o=<disparity_image>] [-p=<point_cloud_file>]\n");
}

static void saveXYZ(const char* filename, const Mat& mat)
{
  const double max_z = 1.0e4;
  FILE* fp = fopen(filename, "wt");
  for (int y = 0; y < mat.rows; y++)
    {
      for (int x = 0; x < mat.cols; x++)
	{
	  Vec3f point = mat.at<Vec3f>(y, x);
	  if (fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z) continue;
	  fprintf(fp, "%f %f %f\n", point[0], point[1], point[2]);
	}
    }
  fclose(fp);
}

int singleDepthMap(Mat img1, Mat img2, Mat img1_colored, Mat img2_colored, Ptr<StereoMatcher> sgbm, 
	int alg, maps m, trinsics &p) {
  namedWindow("left", 1);
  namedWindow("right", 1);
  namedWindow("disparity", 1);
  namedWindow("newPerspective", 1);

 
  Mat disp, disp8, newImage;
 

  getDifferentPerspective(img1, img2, img1_colored, img2_colored, sgbm, m, p, disp, newImage);

  disp.convertTo(disp8, CV_8U);


  imshow("left", img1_colored);
  imshow("right", img2_colored); 
  imshow("disparity", disp8);
  imshow("newPerspective", newImage);
  waitKey();
  return 1;
}


int main(int argc, char** argv)
{
  std::string img1_filename = "testImgs/left.jpg";
  std::string img2_filename = "testImgs/right.jpg";
  
  std::string intrinsic_filename = "testImgs/intrinsics.yml";
  std::string extrinsic_filename = "testImgs/extrinsics.yml";
  std::string disparity_filename = "";
  std::string point_cloud_filename = "";


  int SADWindowSize, numberOfDisparities;
  bool no_display = false;
  float scale = 1;

  int alg = STEREO_BM;
  // STEREO_SGBM;
  // STEREO_HH;
  // STEREO_VAR;
  // STEREO_3WAY;

  numberOfDisparities = 320;
  SADWindowSize = 9;

  if (numberOfDisparities < 1 || numberOfDisparities % 16 != 0)   {
    printf("Command-line parameter error: The max disparity (--maxdisparity=<...>) must be a positive integer divisible by 16\n");
    print_help();
    return -1;
  }
  if (scale < 0) {
    printf("Command-line parameter error: The scale factor (--scale=<...>) must be a positive floating-point number\n");
    return -1;
  }
  if (SADWindowSize < 1 || SADWindowSize % 2 != 1) {
    printf("Command-line parameter error: The block size (--blocksize=<...>) must be a positive odd number\n");
    return -1;
  }

  if ((intrinsic_filename.empty()) || (extrinsic_filename.empty())) {
    printf("Intrinsics and extrinsics necessary\n");
    return -1;
  }

  if (extrinsic_filename.empty() && !point_cloud_filename.empty())    {
    printf("Command-line parameter error: extrinsic and intrinsic parameters must be specified to compute the point cloud\n");
    return -1;
  }

  int color_mode = alg == STEREO_BM ? 0 : -1;

  
  Mat img1_colored = imread(img1_filename, CV_LOAD_IMAGE_COLOR);
  Mat img2_colored = imread(img2_filename, CV_LOAD_IMAGE_COLOR);

  Mat img1 = imread(img1_filename, color_mode);
  Mat img2 = imread(img2_filename, color_mode);
  
 

  Size img_size = img1.size();

  Rect roi1, roi2;

  trinsics p;
  if (!loadCameraParams(intrinsic_filename, extrinsic_filename, p)) {
    printf("Problem loading intrinsics or extrinsics\n");
    return -1;
  }
  stereoRectify(p.M1, p.D1, p.M2, p.D2, img_size, p.R, p.T, p.R1, p.R2, p.P1, p.P2, p.Q,
		CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2);

  maps m;
  initUndistortRectifyMap(p.M1, p.D1, p.R1, p.P1, img_size, CV_16SC2, m.map11, m.map12);
  initUndistortRectifyMap(p.M2, p.D2, p.R2, p.P2, img_size, CV_16SC2, m.map21, m.map22);



  int cn = img1.channels();

  //Mat img1p, img2p, dispp;
  //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
  //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

  int64 t = getTickCount();

  Ptr<StereoMatcher> usedBm;

  if (alg == STEREO_BM)
    usedBm = getStereoBM(roi1, roi2, SADWindowSize, numberOfDisparities);
  else if (alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY)
    usedBm = getStereoSGBM(SADWindowSize, numberOfDisparities, cn, alg);

  t = getTickCount() - t;
  printf("Time elapsed: %fms\n", t * 1000 / getTickFrequency());

  //disp = dispp.colRange(numberOfDisparities, img1p.cols);
  // if (alg != STEREO_VAR)
  //   disp.convertTo(disp8, CV_8U, 255 / (numberOfDisparities*16.));
  // else

  if (!no_display) {
    singleDepthMap(img1, img2, img1_colored, img2_colored, usedBm, alg, m, p);
  }

  // if (!disparity_filename.empty())
  //   imwrite(disparity_filename, disp8);

  // if (!point_cloud_filename.empty())    {
  //     printf("storing the point cloud...");
  //     fflush(stdout);
  //     Mat xyz;
  //     reprojectImageTo3D(disp, xyz, Q, true);
  //     saveXYZ(point_cloud_filename.c_str(), xyz);
  //     printf("\n");
  //   }


  return 0;
}
