#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <quickopencv.h>
#include <iostream>
using namespace cv;
int main()
{
    Mat image = imread("./hqf.jpeg");
    if(image.empty()){
        printf("could not load image...\n");
        return -1;
    }
    namedWindow("输入窗口",WINDOW_FREERATIO);
    imshow("输入窗口",image);

    QuickDemo qd;
    qd.colorSpace_Demo(image,"opentest");

    waitKey(0);
    destroyAllWindows();

    return 0;
}