////////////////////////////////////////////////////////////////////////////////////////
// Implement your own Harris corner detector                                          //
// Use conventional interface tool, such as openCV and Matlab, to read/write images   //
// Implement the function which performs Harris corner detection on an input image    //
////////////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#define ATD at<double>
using namespace std;
using namespace cv;

// Declaration of functions
Mat HarrisConerDetector(Mat input);
Mat getCornerThreshold(Mat input);
void drawingResult (const char* file_name,Mat input, Mat output,Mat hist);

// Declaration of parameters
int blockSize = 3;
int maskSize = 5;
double k = 0.04f;
int thres = 30;

int main()
{
	// Pre-processing
	Mat input = imread("image/input.jpg");
	Mat input_cv = input.clone();
	Mat input_gray;
	cvtColor(input, input_gray, COLOR_BGR2GRAY);
	equalizeHist(input_gray,input_gray);

	// Mathematical algorithm
	Mat output = HarrisConerDetector(input_gray);
	for(int j=0; j<output.rows; j++)
		for(int i=0; i<output.cols; i++) 
			if(output.ATD(j,i)<0) output.ATD(j,i)=0; //neglect negative values of candidates
	normalize(output, output, 0, 255, NORM_MINMAX, CV_8U); //normalize candidates pixel values
	Mat hist = getCornerThreshold(output); //help you to determine threshold by candidates value histogram
	line(hist,Point(thres,0),Point(thres,hist.rows),Scalar(0,255,0),1.5,LINE_AA); //drawing threshold line

	vector<Point2i> corner;	
	for(int j=0; j<output.rows; j++)
		for(int i=0; i<output.cols; i++)
			if (thres < output.at<unsigned char>(j,i)) corner.push_back(Point2i(i,j));
	for(int k=0; k<corner.size(); k++) circle(input, corner[k], 3, Scalar(0,0,255), -1 ); //drawing points over the threshold 
	cvtColor(output, output, COLOR_GRAY2BGR);
	drawingResult("image/Harris_output.jpg",input,output,hist);
	corner.clear();

	// OpenCV algorithm
	Mat output_cv = Mat::zeros(Size(input_gray.cols,input_gray.rows),CV_32FC1);
	cornerHarris(input_gray,output_cv,blockSize-1,maskSize, k); //Harris corner function of OpenCV
	for(int j=0; j<output_cv.rows; j++)
		for(int i=0; i<output_cv.cols; i++) 
			if(output_cv.at<float>(j,i)<0) output_cv.at<float>(j,i)=0; //neglect negative values of candidates
	normalize(output_cv, output_cv, 0, 255, NORM_MINMAX, CV_8U); //normalize candidates pixel values
	Mat hist_cv = getCornerThreshold(output_cv); //help you to determine threshold by candidates value histogram
	line(hist_cv,Point(thres,0),Point(thres,hist_cv.rows),Scalar(0,255,0),1.5,LINE_AA); //drawing threshold line

	vector<Point2i> corner_cv;	
	for(int j=0; j<output_cv.rows; j++)
		for(int i=0; i<output_cv.cols; i++)
			if (thres < output_cv.at<unsigned char>(j,i)) corner_cv.push_back(Point2i(i,j));
	for(int k=0; k<corner_cv.size(); k++) circle(input_cv, corner_cv[k], 3, Scalar(0,0,255), -1 ); //drawing points over the threshold 
	cvtColor(output_cv, output_cv, COLOR_GRAY2BGR);
	drawingResult("image/OpenCV_output.jpg",input_cv,output_cv,hist_cv);
	corner_cv.clear();
	return 0;
}


Mat HarrisConerDetector(Mat input){
	int height = input.rows;
	int width = input.cols;

	// Sobel mask
	Mat sx_mask(blockSize,blockSize,CV_64FC1);
	Mat sy_mask(blockSize,blockSize,CV_64FC1);
	if(blockSize==5){
		sx_mask = (Mat_<double>(5,5) << 2,1,0,-1,-2,
				 		2,1,0,-1,-2,
						4,2,0,-2,-4,
			    			2,1,0,-1,-2,
			    			2,1,0,-1,-2);
		sy_mask = (Mat_<double>(5,5) << 2,2,4,2,2,
				 		1,1,2,1,1,
						0,0,0,0,0,
		    		     	 	-1,-1,-2,-1,-1,
		    		       		-2,-2,-4,-2,-2);
	}	
	else if(blockSize==3){
		sx_mask = (Mat_<double>(3,3) << 1,0,-1,
						2,0,-2,
						1,0,-1);
		sy_mask = (Mat_<double>(3,3) << 1,2,1,
				  		0,0,0,
				      		-1,-2,-1);
	}
	// Derivative matrix
	int intval = floor(blockSize/2);
	Mat dxdx = Mat::zeros(height,width,CV_64FC1);
	Mat dydy = Mat::zeros(height,width,CV_64FC1);
	Mat dxdy = Mat::zeros(height,width,CV_64FC1);
	for(int j=intval; j<height-intval; j++)
	for(int i=intval; i<width-intval; i++)
	{
		double dx=0;
		double dy=0;
		for(int y=-intval; y<=intval; y++)
		for(int x=-intval; x<=intval; x++)
		{
		    dx += (double)input.data[(j+y)*width+(i+x)]*sx_mask.ATD((y+intval),(x+intval));
		    dy += (double)input.data[(j+y)*width+(i+x)]*sy_mask.ATD((y+intval),(x+intval));
		}
		dxdx.ATD(j,i) = dx*dx;
		dydy.ATD(j,i) = dy*dy;
		dxdy.ATD(j,i) = dx*dy;
	}

	// Gaussian mask
	Mat g_mask(maskSize,maskSize,CV_64FC1);
	if(maskSize==5){
		g_mask = (Mat_<double>(5,5) << 1,4,6,4,1,
					 4,16,24,16,4,
					6,24,36,24,6,
				      4,16,24,16,4,
				    1,4,6,4,1);
		g_mask /= 256.0f;
	}
	else if(maskSize==3){
		g_mask = (Mat_<double>(3,3) << 1,2,1,
					   2,4,2,
					1,2,1);
		g_mask /= 16.0f;
	}
	// Weighted matrix
	for (int j=intval; j<height-intval; j++)
	for (int i=intval; i<width-intval; i++)
	{
		double dx2=0;
		double dy2=0;
		double dxy=0;
		for(int y=-intval; y<=intval; y++)
		for(int x=-intval; x<=intval; x++)
		{
		    dx2 += dxdx.ATD((j+y),(i+x))*g_mask.ATD((y+intval),(x+intval));
		    dy2 += dydy.ATD((j+y),(i+x))*g_mask.ATD((y+intval),(x+intval));
		    dy2 += dxdy.ATD((j+y),(i+x))*g_mask.ATD((y+intval),(x+intval));
		}
		dxdx.ATD(j,i) = dx2;
		dydy.ATD(j,i) = dy2;
		dxdy.ATD(j,i) = dxy;
	}

	Mat output(height,width,CV_64FC1);
	for (int j=intval; j<height-intval; j++)
	for (int i=intval; i<width-intval; i++){
		double dx2 = dxdx.ATD(j,i);
		double dy2 = dydy.ATD(j,i);
		double dxy = dxdy.ATD(j,i);
		output.ATD(j,i) = (double)(dx2*dy2-dxy*dxy-k*(dx2+dy2)*(dx2+dy2));
	}
	return output;
}

Mat getCornerThreshold(Mat input){
	// Make Corner candidates space 
	Mat hist = Mat::zeros(Size(256,1),CV_64FC1);
	for(int i=0; i<input.cols*input.rows; i++){
		int loc = (int)input.data[i];
		hist.ATD(0,loc)+=1.0f;
	}
	hist.convertTo(hist,CV_8UC1);
	Mat hist_image = Mat::zeros(Size(256,input.rows),CV_8UC3);
	for(int i=0; i<256; i++) rectangle(hist_image, Rect(i,0,1,(int)(input.rows*(double)hist.data[i]/(double)255)), Scalar(0,0,255), -1);
	flip(hist_image,hist_image,0);
	return hist_image;
}

void drawingResult (const char* file_name, Mat input, Mat output, Mat hist){
	// Combine and Save result images
	Mat drawing = Mat::zeros(Size(input.cols*2+256,input.rows),CV_8UC3);
	ostringstream ss,ss2,ss3;
	ss << "Original Image";
	putText(input, ss.str(), Point(10,20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
	ss2 << "Harris Space";
	putText(output, ss2.str(), Point(10,20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
	ss3 << "Corner Candidates";
	putText(hist, ss3.str(), Point(10,20), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0,255,255), 2);
	vector<Mat> pic;
	pic.push_back(input); input.release();
	pic.push_back(output); output.release();
	pic.push_back(hist); hist.release();
	hconcat(pic,drawing);
	imwrite(file_name, drawing); // exporting image
	pic.clear();
}
