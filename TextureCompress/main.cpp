#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "Block.h"

//   klt
#include "error.h"
#include "base.h"
#include "pnmio.h"
#include "klt.h"

using namespace std;
using namespace cv;


int blockSize = 32;
vector<Block*> blocks;
vector<Vec2b> seedPoints;
vector<Block*> seedBlocks;

uchar* imgRead(const string imgPath, int* ncols, int* nrows);


int RunExample()
{
    uchar* img1, * img2;
    KLT_TrackingContext tc;
    KLT_FeatureList fl;
    int nFeatures = 100;
    int ncols, nrows;
    int i;

    

    tc = KLTCreateTrackingContext();
    //KLTPrintTrackingContext(tc);
    fl = KLTCreateFeatureList(nFeatures);



    img1 = imgRead("..\\Resource\\img1.png", &ncols, &nrows);
    img2 = imgRead("..\\Resource\\img2.png", &ncols, &nrows);


    KLTSelectGoodFeatures(tc, img1, ncols, nrows, fl);

    printf("\nIn first image:\n");
    for (i = 0; i < fl->nFeatures; i++) {
        printf("Feature #%d:  (%f,%f) with value of %d\n",
            i, fl->feature[i]->x, fl->feature[i]->y,
            fl->feature[i]->val);
    } 

    KLTWriteFeatureListToPPM(fl, img1, ncols, nrows, "..\\Resource\\feat1.ppm");
    KLTWriteFeatureList(fl, "feat1.txt", "%3d");

    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);
    KLTTrackFeatures(tc, img1, img2, ncols, nrows, fl);

    printf("\nIn second image:\n");
    for (i = 0; i < fl->nFeatures; i++) {
        printf("Feature #%d:  (%f,%f) with value of %d\n",
            i, fl->feature[i]->x, fl->feature[i]->y,
            fl->feature[i]->val);
    }

    KLTWriteFeatureListToPPM(fl, img2, ncols, nrows, "..\\Resource\\feat2.ppm");
    //KLTWriteFeatureList(fl, "..\\Resource\\feat2.fl", NULL);      /* binary file */
    //KLTWriteFeatureList(fl, "..\\Resource\\feat2.txt", "%5.1f");  /* text file   */

    return 0;
}

int main(int argc, char* argv[]) {


    RunExample();

    // load image
    const string imageName = "..\\Resource\\t10.png";

    Mat img = imread(imageName, 1);
    if (img.empty()) {
        fprintf(stderr, "Can not load image %s\n", imageName);
        return -1;
    }

    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();


    //Mat imgTest(blockSize, blockSize, CV_8UC3);
    //namedWindow("Test");
    //for (int i = 0; i < imgTest.rows; i++)        //遍历每一行每一列并设置其像素值
    //{
    //    for (int j = 0; j < imgTest.cols; j++)
    //    {
    //        imgTest.at<Vec3b>(i, j) = img.at<Vec3b>(blocks[196]->getStartHeight() + i, blocks[196]->getStartWidth() + j);
    //    }
    //}

    //imshow("Test", imgTest);   //窗口中显示图像
    //imwrite("F:\\NewPro\\TextureCompress\\Test\\test.jpg", imgTest);    //保存生成的图片
    //waitKey(5000); //等待5000ms后窗口自动关闭
    //getchar();

    // color histogram simi
    //for (int i = 0; i < seedBlocks.size(); i++)
    //{
    //    int compare_method = 0; //Correlation ( CV_COMP_CORREL )
    //    double simi = compareHist(blocks[0]->getHist(), seedBlocks[i]->getHist(), compare_method);
    //    cout << i << " simi:" << simi << endl;
    //    if (simi > 0.997) {
    //        Vec3f Color = Vec3f(0, 0, 255.0); //bgr 0-255
    //        seedBlocks[i]->setColor(img, Color);
    //    }
    //}
    //imshow("image", img);
    //waitKey();

    //Vec3f Color = Vec3f(0, 0, 255.0); //bgr 0-255
    //blocks[0]->setColor(img, Color);

    ////Match match(2.0f,1.0f,0.0f,5.0f,100.0f,0.0f,-60.0f);
    ////blocks[196]->affineDeformation(img, match);
    ////blocks[196]->rotation(img, match);


    

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
        for (int col = 0; col + blockSize <= img.rows; col += blockSize) {
            Block* tmpBlock = new Block(blockIndex++, blockSize, row, col);
            tmpBlock->computeColorHistogram(img);
            blocks.push_back(tmpBlock);
        }
    }

    //for (int i = 0; i < 9; i++) {
    //    float testTheta = guessTheta(blocks[0]->getHog(), blocks[i]->getHog());

    //    cout << testTheta << endl;
    //}

    // create kit 

    // generate seedPoints
    int seedBlockIndex = 0;
    for (int row = 0; row + blockSize <= img.rows; row += blockSize/4) {
        for (int col = 0; col + blockSize <= img.rows; col += blockSize/4) {
            Block* tmpBlock = new Block(seedBlockIndex++, blockSize, row, col);
            tmpBlock->computeColorHistogram(img);
            seedBlocks.push_back(tmpBlock);
        }
    }

    *ncols = img.cols;
    *nrows = img.rows;
    ptr = (uchar*)malloc((*ncols) * (*nrows) * sizeof(char));
    if (ptr == NULL)
        KLTError("(imgRead) Memory not allocated");

    Mat imgGray;
    cvtColor(img, imgGray, COLOR_BGR2GRAY);

    int index = 0;
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

