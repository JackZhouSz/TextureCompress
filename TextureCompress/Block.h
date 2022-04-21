#ifndef BLOCK_H
#define BLOCK_H

// ===============================
// define image block
// ===============================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "Match.h"
#include "Hog.h"

#define pi 3.14159

using namespace cv;

class Block {
public:
    Block(int _index, int _size, int _startHeight, int _startWidth)
    {
        index = _index;
        size = _size;
        startHeight = _startHeight;
        startWidth = _startWidth;
    }
    int getIndex() { return index; }
    int getStartHeight() { return startHeight; }
    int getStartWidth() { return startWidth; }
    Mat getHist() { return hsvHist; }
    Mat getHog() { return oriHist; }

    void setColor(Mat& img, Vec3f color);
    void affineDeformation(Mat& img, Match match);
    void rotation(Mat& img, Match match);
    void computeColorHistogram(const Mat& img);

private:
    int index;
    int size;
    int startHeight;
    int startWidth;
    Mat hsvHist;
    Mat oriHist;
    //vector<Match> matchList;
};

void Block::setColor(Mat& img, Vec3f color)
{
    for (int row = this->startHeight; row < this->startHeight + this->size; row++) {
        for (int col = this->startWidth; col < this->startWidth + this->size; col++) {
            //img.at<Vec3b>(row, col)[0] = color[0]; //blue
            //img.at<Vec3b>(row, col)[1] = color[1]; //green
            img.at<Vec3b>(row, col)[2] = 50; //red
        }
    }
}

void Block::affineDeformation(Mat& img, Match match)
{
    for (int row = 0; row < this->size; row++) {
        for (int col = 0; col < this->size; col++) {
            int tmpCol = (int)(match.m_00 * col + match.m_01 * row + match.m_a) + this->startWidth;
            int tmpRow = (int)(match.m_10 * col + match.m_11 * row + match.m_b) + this->startHeight;
            if (tmpCol < img.cols && tmpRow < img.rows) {
                img.at<Vec3b>(tmpRow, tmpCol)[0] = 0; //blue
                img.at<Vec3b>(tmpRow, tmpCol)[1] = 255; //green
                img.at<Vec3b>(tmpRow, tmpCol)[2] = 0; //red
            }

        }
    }
}

void Block::rotation(Mat& img, Match match)
{
    for (int row = 0; row < this->size; row++) {
        for (int col = 0; col < this->size; col++) {
            float theta = match.theta / 180.0 * pi;
            int tmpCol = (int)(cos(theta) * col - sin(theta) * row + match.m_a) + this->startWidth;
            int tmpRow = (int)(sin(theta) * col + cos(theta) * row + match.m_b) + this->startHeight;
            if (tmpCol < img.cols && tmpRow < img.rows) {
                img.at<Vec3b>(tmpRow, tmpCol)[0] = 0; //blue
                img.at<Vec3b>(tmpRow, tmpCol)[1] = 255; //green
                img.at<Vec3b>(tmpRow, tmpCol)[2] = 0; //red
            }

        }
    }
}

void Block::computeColorHistogram(const Mat& img)
{
    Mat imgBlock(this->size, this->size, CV_8UC3);

    for (int i = 0; i < imgBlock.rows; i++)
    {
        for (int j = 0; j < imgBlock.cols; j++)
        {
            imgBlock.at<Vec3b>(i, j) = img.at<Vec3b>(this->getStartHeight() + i, this->getStartWidth() + j);
        }
    }

    Mat imgHSV;
    cvtColor(imgBlock, imgHSV, COLOR_BGR2HSV);
    int hBins = 50, sBins = 60;
    int histSize[] = { hBins,sBins };

    // hue varies from 0 to 179, saturation from 0 to 255
    float hRanges[] = { 0, 180 };
    float sRanges[] = { 0, 256 };
    const float* ranges[] = { hRanges, sRanges };

    // Use the 0-th and 1-st channels
    int channels[] = { 0, 1 };
    calcHist(&imgHSV, 1, channels, Mat(), hsvHist, 2, histSize, ranges, true, false);
    normalize(hsvHist, imgHSV, 0, 1, NORM_MINMAX, -1, Mat());

    oriHist = Hog(imgBlock);
}

#endif 

