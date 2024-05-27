#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <quickopencv.h>
#include <iostream>
using namespace cv;
int main(int argc, char** argv)
{
    Mat image = imread("./lsw.jpg");
    if(image.empty()){
       printf("could not load image...\n");
       return -1;
    }
    // namedWindow("skd",WINDOW_AUTOSIZE);
    imshow("skd",image);

    QuickDemo qd;
    qd.key_demo(image);

    waitKey(0);
    destroyAllWindows();
// 119 115 97 100

    return 0;
}