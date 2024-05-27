#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <quickopencv.h>
#include <iostream>
using namespace cv;
int main()
{
    Mat image = imread("./opentest.jpeg");
    if(image.empty()){
       printf("could not load image...\n");
       return -1;
    }
    namedWindow("skd",WINDOW_FREERATIO);
    imshow("skd",image);

    QuickDemo qd;
    qd.mat_creation_demo(image,"opentest");

    waitKey(0);
    destroyAllWindows();

    return 0;
}