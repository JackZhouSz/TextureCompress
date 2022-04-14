#ifndef BLOCK
#define BLOCK

// ===============================
// define image block
// ===============================

#include <opencv2/opencv.hpp>
#include <iostream>
#include "Match.h"

using namespace cv;

class Block {
public:
    Block(int _index,int _size,int _startHeight,int _startWidth)
    {
        index = _index;
        size = _size;
        startHeight = _startHeight;
        startWidth = _startWidth;
    }
    int getIndex() { return index; }
    int getStartHeight() { return startHeight; }
    int getStartWidth() { return startWidth; }

    void setColor(Mat& img, Vec3f color);
    void affineDeformation(Mat& img, Match match);

private:
    int index;
    int size;
    int startHeight;
    int startWidth;
    //vector<Match> matchList;
};

void Block::setColor(Mat& img, Vec3f color)
{
    for (int row = this->startHeight; row < this->startHeight+this->size; row ++) {
        for (int col=this->startWidth; col < this->startWidth + this->size; col++) {
            img.at<Vec3b>(row, col)[0] = color[0]; //blue
            img.at<Vec3b>(row, col)[1] = color[1]; //green
            img.at<Vec3b>(row, col)[2] = color[2]; //red
        }
    }
}

void Block::affineDeformation(Mat& img, Match match)
{
    for (int row = this->startHeight; row < this->startHeight + this->size; row++) {
        for (int col = this->startWidth; col < this->startWidth + this->size; col++) {
            int tmpCol = (int)(match.m_00 * col + match.m_01 * row + match.m_a);
            int tmpRow = (int)(match.m_10 * col + match.m_11 * row + match.m_b);
            if (tmpCol < img.cols && tmpRow < img.rows) {
                img.at<Vec3b>(tmpRow, tmpCol)[0] = 0; //blue
                img.at<Vec3b>(tmpRow, tmpCol)[1] = 255; //green
                img.at<Vec3b>(tmpRow, tmpCol)[2] = 0; //red
            }
            
        }
    }
}

#endif 


