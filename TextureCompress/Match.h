#ifndef MATCH_H
#define MATCH_H

// ===============================
// define affine deformation
// ===============================

#include <opencv2/opencv.hpp>
using namespace cv;

class Match {
public:
	Match(Point2f center,Point2f move,double angle,double scale) {

        angle *= CV_PI / 180;
        double alpha = std::cos(angle) * scale;
        double beta = std::sin(angle) * scale;

        M = cv::Mat::zeros(cv::Size(2, 3), CV_64F);
        double* m = M.ptr<double>();

        Point2f newCenter = center + move;

        //angle 为顺时针旋转角度
        m[0] = alpha;
        m[1] = -beta;
        //m[2] = move.x * alpha - move.y * beta + (1 - alpha) * newCenter.x + beta * newCenter.y;
        m[2] = newCenter.x;
        m[3] = beta;
        m[4] = alpha;
        m[5] = newCenter.y;
        //m[5] = move.y * alpha + move.x * beta - beta * newCenter.x + (1 - alpha) * newCenter.y;


	}
    Mat getMatrix() { return M; }

private:
	Mat M;
	
};


#endif // !MATCH

