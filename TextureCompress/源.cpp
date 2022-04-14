#include <opencv2/opencv.hpp>
#include <iostream>
#include "Block.h"

using namespace std;
using namespace cv;

int blockSize = 32;
vector<Block*> blocks;

int main(int argc, char* argv[]) {
    const char* imagename = "D:\\NewPro\\TextureCompress\\Test\\t2.png";//此处为你自己的图片路径

    //从文件中读入图像
    Mat img = imread(imagename, 1);

    //如果读入图像失败
    if (img.empty()) {
        fprintf(stderr, "Can not load image %s\n", imagename);
        return -1;
    }

    int height = img.rows;
    int width = img.cols;
    int channels = img.channels();

    printf("height=%d,width=%d channels=%d", height, width, channels);

    int blockIndex = 0;
    for (int row = 0; row < height; row+=blockSize) {
        for (int col = 0; col < width; col+=blockSize) {
          /*  img.at<Vec3b>(row, col)[0] = 255;
            img.at<Vec3b>(row, col)[1] = 255;
            img.at<Vec3b>(row, col)[2] = 255;*/
            Block* tmpBlock = new Block(blockIndex++, blockSize, row, col);
            blocks.push_back(tmpBlock);
        }
    }

    Vec3f Color = Vec3f(0, 0, 255.0); //bgr 0-255
    blocks[196]->setColor(img, Color);

    //显示图像
    imshow("image", img);

    //此函数等待按键，按键盘任意键就返回
    waitKey();
    return 0;
}
