#ifndef HOG_H
#define HOG_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

// Parameters
#define N_BINS 36   // Number of bins
#define INF 1e9

// HOG feature

Mat Hog(const Mat& img)
{
    Mat hog = Mat::zeros(1, N_BINS, CV_32FC1);
    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);
    // calculate gradients gx gy
    Mat gx, gy, mag, angle;
    Sobel(imgGray, gx, CV_32F, 1, 0, 1);
    Sobel(imgGray, gy, CV_32F, 0, 1, 1);
    // calculate gradient magnitude and direction
    cartToPolar(gx, gy, mag, angle, 1);
    for (int row = 0; row < img.rows; row++) {
        for (int col = 0; col < img.cols; col++) {
            hog.at<float>(0, angle.at<float>(row, col) / 10) += mag.at<float>(row, col);
        }
    }
    //cout << hog << endl;
    return hog;
}

float guessTheta(const Mat& blockHog, const Mat& seedHog)
{
    int minTheta = 0, tmpTheta = 0;
    float minError = INF, tmpError = 0;
    for (tmpTheta; tmpTheta < N_BINS; tmpTheta++) {
        tmpError = 0;
        for (int i = 0; i < N_BINS; i++) {
            tmpError += pow(blockHog.at<float>(0, i) - seedHog.at<float>(0, ((i + tmpTheta) % N_BINS)), 2);
        }
        if (tmpError < minError) {
            minError = tmpError;
            minTheta = tmpTheta;
        }
    }
    return minTheta * 10;
}



#endif

