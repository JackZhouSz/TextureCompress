#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include <algorithm>
#include "Block.h"

//   klt
#include "error.h"
#include "base.h"
#include "pnmio.h"
#include "klt.h"

using namespace std;
using namespace cv;


int blockSize = 12;
float NMSth = 10.0f;
int matchNum = 0;
vector<Block*> blocks;
vector<Block*> seedBlocks;


const string imgPath = "..\\Resource\\orig1.png";



uchar* imgRead(const string imgPath, int* ncols, int* nrows);

bool myCompare(pair<pair<float, float>, pair<float, int> > a, pair<pair<float, float>, pair<float, int> > b)
{
    return a.second.first < b.second.first;
}

int RunExample()
{
    uchar* img;
    KLT_TrackingContext tc;
    KLT_FeatureList testFl;

    int ncols, nrows;


    img = imgRead(imgPath, &ncols, &nrows);

    tc = KLTCreateTrackingContext();

    testFl = initialAffineTrack(blocks,matchNum);
    myTrackAffine(tc, img, ncols, nrows, testFl);

    vector<vector<pair<pair<float, float>, pair<float,int> > > >NMSlist(blocks.size()), affineList(blocks.size());


    for (int i = 0; i < testFl->nFeatures; i++){
        if (testFl->feature[i]->val != KLT_OOB && testFl->feature[i]->val != KLT_LARGE_RESIDUE) {

            //Apply NMS
            NMSlist[testFl->feature[i]->block_index].push_back(make_pair(make_pair(testFl->feature[i]->aff_x, testFl->feature[i]->aff_y), make_pair(testFl->feature[i]->error, i)));
        }

    }

    //Apply NMS for each Block's matchlist
    for (int i = 0; i < blocks.size(); i++) {
        if (!NMSlist[i].size()) continue;

        sort(NMSlist[i].begin(), NMSlist[i].end(), myCompare);
        affineList[i].push_back(NMSlist[i][0]);
        for (int index = 1; index < NMSlist[i].size(); index++) {
            float minDist = 1e9;
            for (int j = 0; j < affineList[i].size(); j++) {
                float tmpDist = pow(affineList[i][j].first.first - NMSlist[i][index].first.first, 2) +
                    pow(affineList[i][j].first.second - NMSlist[i][index].first.second, 2);
                if (tmpDist < minDist)
                    minDist = tmpDist;
            }
            if (minDist > NMSth) {
                affineList[i].push_back(NMSlist[i][index]);
            }
        }

        for (int j = 0; j < affineList[i].size(); j++) {
            int featureIndex = affineList[i][j].second.second;
            Mat M = Mat::zeros(cv::Size(2, 3), CV_64F);
            double* m = M.ptr<double>();
            m[0] = testFl->feature[featureIndex]->aff_Axx;
            m[1] = testFl->feature[featureIndex]->aff_Axy;
            m[2] = testFl->feature[featureIndex]->aff_x;
            m[3] = testFl->feature[featureIndex]->aff_Ayx;
            m[4] = testFl->feature[featureIndex]->aff_Ayy;
            m[5] = testFl->feature[featureIndex]->aff_y;
            blocks[i]->finalMatchList.push_back(Match(M));
        }
    }
    
    return 0;

}

int main(int argc, char* argv[]) {


    RunExample();

    Mat img = imread(imgPath, 1);
    Mat imgTest(img.rows, img.cols, CV_8UC3);
    namedWindow("Test");

    for (int index = 0; index < blocks.size(); index++) {
        for (int i = 0; i < blocks[index]->finalMatchList.size(); i++) {
            Mat M = blocks[index]->finalMatchList[i].getMatrix();
            double* m = M.ptr<double>();
            for (int row = -blockSize / 2; row < blockSize / 2; row++) {
                for (int col = -blockSize / 2; col < blockSize / 2; col++) {
                    int tmpCol = (int)(m[0] * col + m[1] * row + m[2]);
                    int tmpRow = (int)(m[3] * col + m[4] * row + m[5]);
                    if (tmpCol < img.cols && tmpCol >= 0 && tmpRow < img.rows && tmpRow >= 0) {

                        imgTest.at<Vec3b>(tmpRow, tmpCol) = img.at<Vec3b>(row + blocks[index]->getStartHeight()+ blockSize / 2, col + blocks[index]->getStartWidth() + blockSize / 2);
                    }
                }
            }
        }
    }
    imwrite("..\\Resource\\test.png", imgTest);
    imshow("image", imgTest);
    waitKey();
    
    return 0;
}

uchar* imgRead(const string imgPath, int* ncols, int* nrows)
{
    uchar* ptr;
    Mat img = imread(imgPath, 1);
    if (img.empty()) {
        fprintf(stderr, "Can not load image %s\n", imgPath);
        return NULL;
    }

    // generate blocks
    int blockIndex = 0;
    for (int row = 0; row + blockSize <= img.rows; row += blockSize) {
        for (int col = 0; col + blockSize <= img.cols; col += blockSize) {
            Block* tmpBlock = new Block(blockIndex++, blockSize, row, col);
            tmpBlock->computeColorHistogram(img);
            blocks.push_back(tmpBlock);
        }
    }

    // generate seedPoints
    int seedBlockIndex = 0;
    for (int row = 0; row + blockSize <= img.rows; row += blockSize/4) {
        for (int col = 0; col + blockSize <= img.cols; col += blockSize/4) {
            Block* tmpBlock = new Block(seedBlockIndex++, blockSize, row, col);
            tmpBlock->computeColorHistogram(img);
            seedBlocks.push_back(tmpBlock);
        }
    }


    // color histogram simi
    int index = 2;
    for (int i = 0; i < seedBlocks.size(); i++)
    {
        Mat imgTest = img.clone();
        int compare_method = 0; //Correlation ( CV_COMP_CORREL )
        double simi = compareHist(blocks[index]->getHist(), seedBlocks[i]->getHist(), compare_method);
        //cout << i << " simi:" << simi << endl;
        if (simi>0.5) {
            cout << i << " simi:" << simi << endl;
            float testTheta = guessTheta(blocks[index]->getHog(), seedBlocks[i]->getHog());
            cout << "index "<<i<<" theta "<<testTheta << endl;
            int scale = 1;
            Point2f move = Point2f(seedBlocks[i]->getStartWidth() - blocks[index]->getStartWidth(),
                seedBlocks[i]->getStartHeight() - blocks[index]->getStartHeight());
            blocks[index]->addInitMatch(move, testTheta, scale);
            matchNum++;
        }
    }


    *ncols = img.cols;
    *nrows = img.rows;
    ptr = (uchar*)malloc((*ncols) * (*nrows) * sizeof(char));
    if (ptr == NULL)
        KLTError("(imgRead) Memory not allocated");

    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    uchar* tmpptr = ptr;
    for (int i = 0; i < *nrows; i++)
    {
        for (int j = 0; j < *ncols; j++)
        {
            *tmpptr = imgGray.at<uchar>(i, j);
            tmpptr++;
        }
    }
    return ptr;
}

