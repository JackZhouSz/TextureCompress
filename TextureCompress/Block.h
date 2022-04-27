#ifndef BLOCK_H
#define BLOCK_H

// ===============================
// define image block
// ===============================

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "Match.h"

#define pi 3.14159

using namespace cv;
using namespace std;

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
    int getSize() { return size; }
    int getStartHeight() { return startHeight; }
    int getStartWidth() { return startWidth; }
    Mat getHist() { return hsvHist; }
    Mat getHog() { return oriHist; }
    Match getMatch(int index) { return matchList[index]; }

    void setColor(Mat& img, Vec3f color);
    void affineDeformation(Mat& img, Match match);
    void computeColorHistogram(const Mat& img);
    void addMatch(Point2f move, double angle, double scale);

private:
    int index;
    int size;
    int startHeight;
    int startWidth;
    Mat hsvHist;
    Mat oriHist;
    vector<Match> matchList;
};

Mat Hog(const Mat& img);

float guessTheta(const Mat& blockHog, const Mat& seedHog);

#endif 

