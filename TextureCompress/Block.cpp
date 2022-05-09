#include "Block.h"

// HOG feature
// Parameters
#define N_BINS 36   // Number of bins
#define INF 1e9

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

void Block::setColor(Mat& img, Vec3f color)
{
    for (int row = this->startHeight; row < this->startHeight + this->size; row++) {
        for (int col = this->startWidth; col < this->startWidth + this->size; col++) {
            //img.at<Vec3b>(row, col)[0] = color[0]; //blue
            //img.at<Vec3b>(row, col)[1] = color[1]; //green
            img.at<Vec3b>(row, col)[2] = 200; //red
        }
    }
}

void Block::addInitMatch(Point2f move, double angle, double scale)
{
    //Match(Point2f center, Point2f move, double angle, double scale) 
    Point2f center= Point2f(this->getStartWidth() + this->getSize() * 1.0 / 2, this->getStartHeight() + this->getSize() * 1.0 / 2);
    Match m(center, move, angle, scale);
    this->initMatchList.push_back(m);
}

void Block::affineDeformation(Mat& img, Match match)
{
    Mat M = match.getMatrix();
    double* m = M.ptr<double>();

    for (int row = -this->size / 2; row < this->size / 2; row++) {
        for (int col = -this->size / 2; col < this->size / 2; col++) {
            int tmpCol = (int)(m[0] * col + m[1] * row + m[2]);
            int tmpRow = (int)(m[3] * col + m[4] * row + m[5]);
            if (tmpCol < img.cols && tmpCol >= 0 && tmpRow < img.rows && tmpRow >= 0) {
                //img.at<Vec3b>(tmpRow, tmpCol)[0] = 0; //blue
                img.at<Vec3b>(tmpRow, tmpCol)[1] = 200; //green
                //img.at<Vec3b>(tmpRow, tmpCol)[2] = 0; //red
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

    Mat imgHSV,imgGray;
    Mat mean, stddev;
    cvtColor(imgBlock, imgGray, COLOR_BGR2GRAY);
    meanStdDev(imgGray, mean, stddev);
    this->meanLight = mean.at<double>(0, 0);
    this->stddev = stddev.at<double>(0, 0);

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

