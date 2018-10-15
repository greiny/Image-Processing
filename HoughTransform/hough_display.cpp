#include <cmath>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <sstream>
#include <pthread.h>
#include <cstdio>
#include <chrono>
#include <unistd.h>

#include "opencv2/core.hpp"
#include <opencv2/core/utility.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace std;
using namespace cv;

void* CPU_thread(void *);

// constans of Gaussian blur
Size ksize = Size(5, 5);
double sigma1 = 2;
double sigma2 = 2;

// constans of Hough circles
float dp = 1;  //image resize
float minDist=10;
int cannythres=20;
int votesthres=30;
int minRad=1;
int maxRad=100;
int R = 42;

int main(int argc, const char* argv[])
{
	int status_cpu = 0;
	pthread_t CPU_thread_id;
	if(pthread_create(&CPU_thread_id,NULL,CPU_thread,NULL)) cout << "cpu_thread create error" << endl;
	pthread_join(CPU_thread_id,(void**)&status_cpu);
    return 0;
}

void* CPU_thread(void *)
{
	bool flag=1;
	Mat frame, src, dst_cpu, mask;
	// Time checking start
	int frames = 0;
	float time = 0, fps = 0;
   	auto t0 = std::chrono::high_resolution_clock::now();

	frame = imread("image/test.png");
	cv::cvtColor(frame, src, COLOR_BGR2GRAY);
	cv::cvtColor(src, dst_cpu, COLOR_GRAY2BGR);
	// Reduce the noise so we avoid false circle detection
	GaussianBlur( src, mask, ksize, sigma1, sigma2 );
	Canny(mask, mask, 100, 200, 3); // it can be proceed only at cpu
	// Apply the Hough Transform to find the circles
	Mat accumulator(mask.rows+R,mask.cols+R,CV_32FC1);
	accumulator = Scalar(0);
	for (int y=0; y<mask.rows; y++)
	{
	    for (int x=0; x<mask.cols; x++)
	    {
		// if edge pixek is found
		if (mask.data[y+x*mask.cols] == 255)
		{
		    for (int d=0 ; d<360 ; d++)
		    {
				int a = x-floor(R*cos(d*M_PI/180));
		       		int b = y-floor(R*sin(d*M_PI/180));
				accumulator.data[(b+R)+(a+R)*accumulator.cols] += 10;
		    }
		}
	    }
	}
	Mat show(accumulator.rows,accumulator.cols,CV_8UC1);
	double minVal, maxVal;
	minMaxLoc(accumulator, &minVal, &maxVal); //find minimum and maximum intensities
	for(int ii=0; ii < accumulator.rows*accumulator.cols; ii++)
	{
		show.data[ii] = (unsigned char)accumulator.data[ii];
	}
	cv::cvtColor(show, show, COLOR_GRAY2BGR);


	vector<Vec3f> circles_cpu;
	HoughCircles( mask, circles_cpu, CV_HOUGH_GRADIENT, dp, minDist, cannythres, votesthres, minRad, maxRad );   
	// Draw the circles detected
	for( size_t i = 0; i < circles_cpu.size(); i++ )
	{
		Point center(cvRound(circles_cpu[i][0]), cvRound(circles_cpu[i][1]));
		int radius = cvRound(circles_cpu[i][2]);  
		circle( dst_cpu, center, radius, Scalar(0,0,255), 3, 8, 0 );// circle outline
	}
	imwrite("image/hough.png", show);
	imwrite("image/final.png", dst_cpu);
	return 0;
}

