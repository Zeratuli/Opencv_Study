#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <quickopencv.h>
#include <iostream>
using namespace cv;
int main(int argc, char** argv)
{
    Mat image = imread("./image/fat-cat.jpg");
    if(image.empty()){
       printf("could not load image...\n");
       return -1;
    }
    namedWindow("skd",WINDOW_AUTOSIZE);
    //namedWindow("skd",WINDOW_FREERATIO);
    imshow("skd",image);

    QuickDemo qd;
    qd.color_style_demo(image);

    waitKey(0);
    destroyAllWindows();

    return 0;
}