#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <cmath>
#include <algorithm>
#include <Windows.h>
#include "Block.h"

//   klt
#include "error.h"
#include "base.h"
#include "pnmio.h"
#include "klt.h"

using namespace std;
using namespace cv;

int threadNum = 16;
int blockSize = 12;
int ncols, nrows;


KLT_FeatureList testFl[16] = { nullptr };

float NMSth = 50.0f;
float simiThread = 0.5;
float equalBlockThread = 1000;
int matchNum = 0;
vector<Block*> blocks;
vector<Block*> seedBlocks;

mutex mtx; // protect img

const string imgPath = "..\\Resource\\orig.png";

uchar* imgRead(const string imgPath, int* ncols, int* nrows);

bool myCompare(pair<pair<float, float>, pair<float, int> > a, pair<pair<float, float>, pair<float, int> > b)
{
    return a.second.first < b.second.first;
}

void FindInitMatch(int start, int end) {
    // color histogram simi
    for (int index = start; index < end; index++) {

        printf("\rcompute block simi[%.2f%%]", (index-start) * 100.0 / (end-start));

        if (blocks[index]->equalBlock != -1) {
            int scale = 1;
            Point2f move = Point2f(0, 0);
            blocks[index]->addInitMatch(move, 0, scale);
            continue;
        }

        for (int i = 0; i < seedBlocks.size(); i++)
        {
            int compare_method = 0; //Correlation ( CV_COMP_CORREL )
            double simi = compareHist(blocks[index]->getHist(), seedBlocks[i]->getHist(), compare_method);
            //cout << i << " simi:" << simi << endl;
            if (simi > simiThread) {
                //cout << i << " simi:" << simi << endl;
                float testTheta = guessTheta(blocks[index]->getHog(), seedBlocks[i]->getHog());
                //cout << "index " << i << " theta " << testTheta << endl;
                int scale = 1;
                Point2f move = Point2f(seedBlocks[i]->getStartWidth() - blocks[index]->getStartWidth(),
                    seedBlocks[i]->getStartHeight() - blocks[index]->getStartHeight());
                blocks[index]->addInitMatch(move, testTheta, scale);
                matchNum++;
            }
        }
    }
}

void FindingSimi(int start, int end, uchar* img, int threadIndex)
{
    KLT_TrackingContext tc;

    int tmpMatchNum = 0;
    mtx.lock();
    vector<Block*>::const_iterator first = blocks.begin() + start;
    vector<Block*>::const_iterator last = blocks.begin() + end;
    vector<Block*> tmpBlocks(first, last);
    for (vector<Block*>::iterator it = tmpBlocks.begin(); it != tmpBlocks.end(); it++) {
        tmpMatchNum += (*it)->initMatchList.size();
    }
    mtx.unlock();

    tc = KLTCreateTrackingContext();
    testFl[threadIndex] = initialAffineTrack(tmpBlocks, tmpMatchNum, start);
    for (vector<Block*>::const_iterator it = first; it != last; it++) {
        (*it)->initMatchList.clear();
    }
    myTrackAffine(tc, img, ncols, nrows, testFl[threadIndex]);

    return ;

}

void CreatingCharts()
{
    vector<int> matchRecord(blocks.size(), 1e9);
    vector<set<pair<int, int> > >tmpCover(blocks.size()), inverseCover(blocks.size()); // blockIndex & matchIndex
    vector<set<int> >candidateRegion(blocks.size()); // blockIndex & matchIndex

    Mat img = imread(imgPath, 1);
    int colBlockNum = img.cols / blockSize;
    int rowBlockNum = img.rows / blockSize;

    // compute inverseCover
    for (int index = 0; index < blocks.size(); index++) {
        printf("\r compute inverseCover [%.2f%%]", index * 100.0 / blocks.size());
        for (int i = 0; i < blocks[index]->finalMatchList.size(); i++) {
            Mat M = blocks[index]->finalMatchList[i].getMatrix();
            double* m = M.ptr<double>();
            for (int row = -blockSize / 2; row < blockSize / 2; row++) {
                for (int col = -blockSize / 2; col < blockSize / 2; col++) {
                    int tmpCol = (int)(m[0] * col + m[1] * row + m[2]);
                    int tmpRow = (int)(m[3] * col + m[4] * row + m[5]);
                    int coverIndex = tmpRow / blockSize * colBlockNum + tmpCol / blockSize;
                    if (coverIndex < tmpCover.size())
                        tmpCover[coverIndex].insert(make_pair(index, i));
                }
            }
        }
    }

    // make sure one block's match show only once
    for (int index = 0; index < blocks.size(); index++) {
        int preIndex = -1;
        for (set<pair<int, int> >::iterator it = tmpCover[index].begin(); it != tmpCover[index].end(); it++) {
            if (it->first == preIndex)
                continue;
            else {
                preIndex = it->first;
                inverseCover[index].insert(*it);
            }
        }
    }
    // compute candidateRegion
    for (int index = 0; index < blocks.size(); index++) {
        printf("\r compute candidateRegion [%.2f%%]", index * 100.0 / blocks.size());
        for (set<pair<int, int> >::iterator it = inverseCover[index].begin(); it != inverseCover[index].end(); it++) {
            
            Mat M = blocks[it->first]->finalMatchList[it->second].getMatrix();
            double* m = M.ptr<double>();
            for (int row = -blockSize / 2; row < blockSize / 2; row++) {
                for (int col = -blockSize / 2; col < blockSize / 2; col++) {
                    int tmpCol = (int)(m[0] * col + m[1] * row + m[2]);
                    int tmpRow = (int)(m[3] * col + m[4] * row + m[5]);
                    int coverIndex = tmpRow / blockSize * (img.cols / blockSize) + tmpCol / blockSize;
                    candidateRegion[index].insert(coverIndex);
                }
            }
        }
    }

    set<int> Ie, epitome;
    vector<set<int> > chartSet;
    vector<int> flag(blocks.size(), 0);
    while (Ie.size() != blocks.size()) {
        printf("\r compute epitome [%.2f%%]", Ie.size() * 100.0 / blocks.size());
        set<int> nowChart, nextChart;
        int maxCoverIndex = 0, maxTmp = 0;
        for (int index = 0; index < blocks.size(); index++) {
            if (Ie.find(index) == Ie.end()) {
                int count = 0;
                for (set<pair<int, int> >::iterator it = inverseCover[index].begin(); it != inverseCover[index].end(); it++) {
                    if (Ie.find(it->first) == Ie.end()) count++;
                }
                if (count > maxTmp) {
                    maxTmp = count;
                    maxCoverIndex = index;
                }
            }
        }
        int cost = 0;
        for (set<int>::iterator it = candidateRegion[maxCoverIndex].begin(); it != candidateRegion[maxCoverIndex].end(); it++) {
            if (epitome.find(*it) == epitome.end()) {
                cost++;
            }
        }
        if (cost > maxTmp) {
            Ie.insert(maxCoverIndex);
            epitome.insert(maxCoverIndex);
            nowChart.insert(maxCoverIndex);
            chartSet.push_back(nowChart);
            //matchRecord[maxCoverIndex] = 0;
        }
        else {
            for (set<int>::iterator it = candidateRegion[maxCoverIndex].begin(); it != candidateRegion[maxCoverIndex].end(); it++) {
                Ie.insert(*it);
                epitome.insert(*it);
                nowChart.insert(*it);
                //matchRecord[*it] = 0;
            }

            for (set<pair<int, int> >::iterator it = inverseCover[maxCoverIndex].begin(); it != inverseCover[maxCoverIndex].end(); it++) {
                Ie.insert(it->first);
                if (it->second < matchRecord[it->first])
                    matchRecord[it->first] = it->second;
            }

            nextChart = nowChart;
            // growth chart with block's candidates
            // search blocks inside or adjacent to current chart
            for (set<int>::iterator it = nowChart.begin(); it != nowChart.end(); it++) {
                // Check Boarder
                vector<int> tmpIndex;
                if (flag[*it] == 0)
                    tmpIndex.push_back(*it);
                int j = *it / colBlockNum, i = *it - j * colBlockNum;
                if (i >= 1 && flag[*it - 1] == 0)
                    tmpIndex.push_back(*it - 1);
                if (i < colBlockNum - 1 && flag[*it + 1] == 0)
                    tmpIndex.push_back(*it + 1);
                if (j >= 1 && flag[*it - colBlockNum] == 0)
                    tmpIndex.push_back(*it - colBlockNum);
                if (j < rowBlockNum - 1 && flag[*it + colBlockNum] == 0)
                    tmpIndex.push_back(*it + colBlockNum);

                for (int i = 0; i < tmpIndex.size(); i++)
                {
                    int a = tmpIndex[i];
                    flag[a] = 1;
                    int cost = 0, benefit = 0;
                    for (set<int>::iterator it = candidateRegion[a].begin(); it != candidateRegion[a].end(); it++) {
                        if (epitome.find(*it) == epitome.end()) {
                            cost++;
                        }

                    }
                    for (set<pair<int, int> >::iterator it = inverseCover[a].begin(); it != inverseCover[a].end(); it++) {
                        if (Ie.find(it->first) == Ie.end()) {
                            benefit++;
                        }
                    }

                    //Ҫ������ĳ��block��canndiate j������Ķ���Cj����С�ڿ��Ը���ѹ���Ŀ���
                    if (benefit > cost) {
                        for (set<int>::iterator it = candidateRegion[a].begin(); it != candidateRegion[a].end(); it++) {
                            Ie.insert(*it);
                            epitome.insert(*it);
                            nextChart.insert(*it);
                            //matchRecord[*it] = 0;
                        }
                        for (set<pair<int, int> >::iterator it = inverseCover[a].begin(); it != inverseCover[a].end(); it++) {
                            Ie.insert(it->first);
                            if (it->second < matchRecord[it->first])
                                matchRecord[it->first] = it->second;
                        }
                    }
                }
            }
            nowChart = nextChart;
            chartSet.push_back(nowChart);
        }
        
    }
    
    // Optimizing the transform map
    for (int index = 0; index < blocks.size(); index++) {
        for (int i = 0; i < blocks[index]->finalMatchList.size() && i < matchRecord[index]; i++) {
            Mat M = blocks[index]->finalMatchList[i].getMatrix();
            double* m = M.ptr<double>();
            bool flag = true;
            for (int row = -blockSize / 2; row < blockSize / 2; row++) {
                for (int col = -blockSize / 2; col < blockSize / 2; col++) {
                    int tmpCol = (int)(m[0] * col + m[1] * row + m[2]);
                    int tmpRow = (int)(m[3] * col + m[4] * row + m[5]);
                    int coverIndex = tmpRow / blockSize * colBlockNum + tmpCol / blockSize;
                    if (epitome.find(coverIndex) == epitome.end()) {
                        flag = false;
                        break;
                    }
                }
                if (flag == false) break;
            }
            if (flag == true) {
                matchRecord[index] = i;
                break;
            }
        }
    }

    // Reconstructed Test
    Mat imgTest(img.rows, img.cols, CV_8UC3);
    namedWindow("Test");

    for (set<int>::iterator it = epitome.begin(); it != epitome.end(); it++) {
        for (int row = -blockSize / 2; row < blockSize / 2; row++) {
            for (int col = -blockSize / 2; col < blockSize / 2; col++) {
                imgTest.at<Vec3b>(row + blocks[*it]->getStartHeight() + blockSize / 2, col + blocks[*it]->getStartWidth() + blockSize / 2) = img.at<Vec3b>(row + blocks[*it]->getStartHeight() + blockSize / 2, col + blocks[*it]->getStartWidth() + blockSize / 2);
            }
        }
    }

    imwrite("..\\Resource\\test.png", imgTest);
    imshow("image", imgTest);
    waitKey();

    // Reconstructed Test
    Mat imgRecon(img.rows, img.cols, CV_8UC3);
    namedWindow("ReconstructedTest");

    for (int i = 0; i < blocks.size(); i++) {
        Mat M = blocks[i]->finalMatchList[matchRecord[i]].getMatrix();
        double* m = M.ptr<double>();
        for (int row = -blockSize / 2; row < blockSize / 2; row++) {
            for (int col = -blockSize / 2; col < blockSize / 2; col++) {
                int tmpCol = (int)(m[0] * col + m[1] * row + m[2]);
                int tmpRow = (int)(m[3] * col + m[4] * row + m[5]);
                if (tmpCol < img.cols && tmpCol >= 0 && tmpRow < img.rows && tmpRow >= 0) {
                    imgRecon.at<Vec3b>(row + blocks[i]->getStartHeight() + blockSize / 2, 
                        col + blocks[i]->getStartWidth() + blockSize / 2) = img.at<Vec3b>(tmpRow, tmpCol);
                }

            }
        }
    }
    imwrite("..\\Resource\\imgRecon.png", imgRecon);
    imshow("imgRecon", imgRecon);
    waitKey();


    return;
}

int main(int argc, char* argv[]) {

    
    uchar* initImg = imgRead(imgPath, &ncols, &nrows);

    vector<uchar*> img(threadNum);
    for (int index = 0; index < threadNum; index++) {
        img[index] = (uchar*)malloc((ncols) * (nrows) * sizeof(char));
        // copy img content
        int i;
        for (i = 0; i < (ncols) * (nrows); i++) {
            *(img[index] + i) = *(initImg + i);
        }
    }

    vector<thread> t(threadNum);
    // why more threads wrong?
    threadNum = 16;
    for (int i = 0; i < threadNum; i++) {
        t[i] = thread(FindingSimi, i * blocks.size() / threadNum, (i + 1) * blocks.size() / threadNum, img[i], i);
    }
    for (int i = 0; i < threadNum; i++) {
        t[i].join();
    }

    vector<vector<pair<pair<float, float>, pair<float, int> > > >NMSlist(blocks.size()), affineList(blocks.size());

    for (int threadIndex = 0; threadIndex < threadNum; threadIndex++) {
        for (int i = 0; i < testFl[threadIndex]->nFeatures; i++) {
            if (testFl[threadIndex]->feature[i]->val == KLT_TRACKED) {
                // Init NMS
                NMSlist[testFl[threadIndex]->feature[i]->block_index].push_back(
                    make_pair(make_pair(testFl[threadIndex]->feature[i]->aff_x, 
                        testFl[threadIndex]->feature[i]->aff_y), 
                    make_pair(testFl[threadIndex]->feature[i]->error, i)));
            }
        }

        //Apply NMS for each Block's matchlist
        for (int i = threadIndex * blocks.size() / threadNum; i < (threadIndex + 1) * blocks.size() / threadNum; i++) {
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
                m[0] = testFl[threadIndex]->feature[featureIndex]->aff_Axx;
                m[1] = testFl[threadIndex]->feature[featureIndex]->aff_Axy;
                m[2] = testFl[threadIndex]->feature[featureIndex]->aff_x;
                m[3] = testFl[threadIndex]->feature[featureIndex]->aff_Ayx;
                m[4] = testFl[threadIndex]->feature[featureIndex]->aff_Ayy;
                m[5] = testFl[threadIndex]->feature[featureIndex]->aff_y;
                blocks[i]->finalMatchList.push_back(Match(M));
            }
        }
    }

    for (int i = 0; i < blocks.size(); i++) {
        if (blocks[i]->equalBlock != -1) {
            for (vector<Match>::iterator it = blocks[blocks[i]->equalBlock]->finalMatchList.begin(); it != blocks[blocks[i]->equalBlock]->finalMatchList.end(); it++)
                blocks[i]->finalMatchList.push_back(*it);
        }
    }

    // Reconstructed Test
    /*Mat img = imread(imgPath, 1);
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
            imshow("image", imgTest);
            waitKey();
        }
    }
    imwrite("..\\Resource\\test.png", imgTest);
    imshow("image", imgTest);
    waitKey();*/

    CreatingCharts();
    
    return 0;
}


float computeDiff(int index1, int index2, const Mat& img)
{
    float diff = 0;
    int height1 = blocks[index1]->getStartHeight();
    int height2 = blocks[index2]->getStartHeight();
    int width1 = blocks[index1]->getStartWidth();
    int width2 = blocks[index2]->getStartWidth();

    for (int i = 0; i < blockSize; i++){
        for (int j = 0; j < blockSize; j++){
            // BGR
            float t10 = float(img.at<Vec3b>(height1 + i, width1 + j)[0]);
            float t11 = float(img.at<Vec3b>(height1 + i, width1 + j)[1]);
            float t12 = float(img.at<Vec3b>(height1 + i, width1 + j)[2]);
            float t20 = float(img.at<Vec3b>(height2 + i, width2 + j)[0]);
            float t21 = float(img.at<Vec3b>(height2 + i, width2 + j)[1]);
            float t22 = float(img.at<Vec3b>(height2 + i, width2 + j)[2]);

            diff += abs(t10 - t20);
            diff += abs(t11 - t21);
            diff += abs(t12 - t22);
        }
    }
    return diff;
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

    // compute eualBlocks
    for (int i = 0; i < blocks.size(); i++) {
        printf("\rcompute eualBlocks[%.2f%%]", i * 100.0 / (blocks.size() - 1));
        for (int j = 0; j < i; j++) {
            if (blocks[j]->equalBlock != -1) continue;
            if (computeDiff(i, j, img) < equalBlockThread)
            {
                blocks[i]->equalBlock = j;
                break;
            }

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

    // accelerate using thread
    vector<thread> t(threadNum);
    for (int i = 0; i < threadNum; i++) {
        t[i] = thread(FindInitMatch, i * blocks.size() / threadNum, (i + 1) * blocks.size() / threadNum);
    }
    for (int i = 0; i < threadNum; i++) {
        t[i].join();
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

