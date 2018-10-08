#include <iostream>
#include <stdlib.h>
#include <vector>
#include <unordered_map>
#include <string>
#include <iostream>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

int* makeHist(Mat input);
int* makeAccum(int* hist);
Mat drawHist(Mat input, const int* hist, const int* accum);

int main()
{
	// importing image as grayscale
	Mat input = imread("image/input.jpg",0);
	int* hist_input = makeHist(input);
	int* accum_input = makeAccum(hist_input);
	Mat draw_input = drawHist(input,hist_input,accum_input);
	imwrite("image/draw_input.jpg", draw_input); // exporting image
	draw_input.release();

	// equalize image 
	int* hist_output = new int[256];
	for(int k=0; k<256; k++) hist_output[k]=0;
	Mat output = Mat::zeros(Size(input.cols,input.rows),CV_8UC1);
	for(int i=0; i<output.rows; i++){
		for(int j=0; j<output.cols; j++){
			output.data[input.cols*i+j] = (unsigned char)((double)accum_input[input.data[input.cols*i+j]]*255/(input.cols*input.rows));
			int loc = (int)output.data[input.cols*i+j];
			hist_output[loc]++;
		}
	}
	int* accum_output = makeAccum(hist_output);
	Mat draw_output = drawHist(output, hist_output, accum_output);
	imwrite("image/output.jpg", draw_output); // exporting image
	draw_output.release();

	// comparing with OpenCV result
	Mat output_cv;
	equalizeHist(input,output_cv);
	int* hist_cv = makeHist(output_cv);
	int* accum_cv = makeAccum(hist_cv);
	Mat draw_cv = drawHist(output_cv,hist_cv,accum_cv);
	imwrite("image/output_cv.jpg", draw_cv); // exporting image
	draw_cv.release();

    return 0;
}

int* makeHist(Mat input){ 
	int* hist =  new int[256]; // save pixel value to 1D
	for(int k=0; k<256; k++) hist[k]=0;
	for(int i=0; i<input.cols*input.rows; i++){
		int loc = (int)input.data[i];
		hist[loc]++;
	} 
	return hist;
}

int* makeAccum(int* hist){
	// accumulating histogram information
	int* sum = new int[256];
	for(int i=0; i<256; i++){
		sum[i]=0; // initialization
		for(int j=0; j<=i; j++) sum[i] += (int)hist[j];
	}
	return sum;
}

Mat drawHist(Mat input, const int* hist, const int* accum){
	vector<Mat> images(3);
	Mat drawing = Mat::zeros(Size(input.cols*3,input.rows),CV_8UC3);
	cvtColor(input,input,COLOR_GRAY2BGR);
	input.copyTo(images[0]);

	// drawing histogram
	int max=0;
	for(int i=0; i<256; i++) if(max<(int)hist[i]) max = (int)hist[i];
	Mat hist_image = Mat::zeros(Size(256,input.rows),CV_8UC3);
	for(int i=0; i<256; i++) rectangle(hist_image, Rect(i,0,1,(int)(input.rows*hist[i]/max)), Scalar(0,0,255), -1);
	resize(hist_image,hist_image,Size(input.cols,input.rows));
	flip(hist_image,hist_image,0); // flip the image with x axis for erect image
	hist_image.copyTo(images[1]);
	hist_image.release();

	// drawing accumulated histogram
	Mat accum_image = Mat::zeros(Size(256,input.rows),CV_8UC3);
	for(int i=0; i<256; i++) rectangle(accum_image, Rect(i,0,1,(int)(accum[i]/input.cols)), Scalar(0,0,255), -1); // drawing sum of histogram with scaling
	resize(accum_image,accum_image,Size(input.cols,input.rows));
	flip(accum_image,accum_image,0); // flip the image with x axis for erect image
	accum_image.copyTo(images[2]);
	accum_image.release();

	// combining images
	hconcat(images,drawing);
	images.clear();
	return drawing;
}
