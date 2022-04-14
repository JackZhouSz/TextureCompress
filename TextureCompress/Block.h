#ifndef BLOCK
#define BLOCK

#include <opencv2/opencv.hpp>
#include <iostream>

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

private:
    int index;
    int size;
    int startHeight;
    int startWidth;
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

#endif 


