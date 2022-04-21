#include <opencv2/opencv.hpp>
#include <iostream>
#include <cmath>
#include "Block.h"

using namespace std;
using namespace cv;

int blockSize = 32;
vector<Block*> blocks;
vector<Vec2b> seedPoints;
vector<Block*> seedBlocks;


int main(int argc, char* argv[]) {

    // load image
    const string imageName = "F:\\NewPro\\TextureCompress\\Test\\t10.png";

    Mat img = imread(imageName, 1);
    if (img.empty()) {
        fprintf(stderr, "Can not load image %s\n", imageName);
        return -1;
    }

    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();

    // generate blocks
    int blockIndex = 0;
    for (int row = 0; row + blockSize <= height; row += blockSize) {
        for (int col = 0; col + blockSize <= width; col += blockSize) {
            Block* tmpBlock = new Block(blockIndex++, blockSize, row, col);
            tmpBlock->computeColorHistogram(img);
            blocks.push_back(tmpBlock);
        }
    }

    for (int i = 0; i < 9; i++) {
        float testTheta = guessTheta(blocks[0]->getHog(), blocks[i]->getHog());

        cout << testTheta << endl;
    }
    

    // generate seedPoints
    //int seedBlockIndex = 0;
    //for (int row = 0; row + blockSize <= height; row += blockSize/4) {
    //    for (int col = 0; col + blockSize <= width; col += blockSize/4) {
    //        Block* tmpBlock = new Block(seedBlockIndex++, blockSize, row, col);
    //        tmpBlock->computeColorHistogram(img);
    //        seedBlocks.push_back(tmpBlock);
    //    }
    //}


    

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




//#include<opencv2/opencv.hpp>
//#include<math.h>
//using namespace cv;
//
//Mat src, gray, dst, harrisRspImg;
//double harrisMinRsp;
//double harrisMaxRsp;
//int qualityLevel = 30;
//int maxCount = 100;
//void cornerTrack(int, void*);
//
//int main()
//{
//    src = imread("F:\\NewPro\\TextureCompress\\Test\\t2.png");
//    if (src.empty())
//    {
//        printf("can not load image \n");
//        return -1;
//    }
//    namedWindow("input", WINDOW_AUTOSIZE);
//    imshow("input", src);
//    cvtColor(src, gray, COLOR_BGR2GRAY);
//
//    float k = 0.04;
//    dst = Mat::zeros(src.size(), CV_32FC(6));
//    harrisRspImg = Mat::zeros(src.size(), CV_32FC1);
//    //计算角点检测的图像块的特征值和特征向量
//    cornerEigenValsAndVecs(gray, dst, 3, 3, 4);
//    for (int r = 0; r < dst.rows; r++)
//    {
//        for (int c = 0; c < dst.cols; c++)
//        {
//            double lambda1 = dst.at<Vec6f>(r, c)[0];
//            double lambda2 = dst.at<Vec6f>(r, c)[1];
//            harrisRspImg.at<float>(r, c) = lambda1 * lambda2 - k * pow((lambda1 + lambda2), 2);
//
//        }
//    }
//    minMaxLoc(harrisRspImg, &harrisMinRsp, &harrisMinRsp, 0, 0, Mat());
//
//    namedWindow("output", WINDOW_AUTOSIZE);
//    createTrackbar("QualityValue", "output", &qualityLevel, maxCount, cornerTrack);
//    cornerTrack(0, 0);
//    waitKey(0);
//    return 0;
//}
//
//void cornerTrack(int, void*)
//{
//    if (qualityLevel < 10)
//    {
//        qualityLevel = 10;
//    }
//    Mat showImage = src.clone();
//    float t = harrisMinRsp + ((((double)qualityLevel) / maxCount) * (harrisMaxRsp - harrisMinRsp));
//    for (int r = 0; r < src.rows; r++)
//    {
//        for (int c = 0; c < src.cols; c++)
//        {
//            float value = harrisRspImg.at<float>(r, c);
//            if (value > t)
//            {
//                circle(showImage, Point(c, r), 2, Scalar(0, 255, 255), 2, 8, 0);
//            }
//        }
//    }
//    imshow("output", showImage);
//}

