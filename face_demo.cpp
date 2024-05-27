#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;

int main(void){
    std::string pb_file_path = "./facetest/face_detector/opencv_face_detector_uint8.pb";
    std::string pbtxt_file_path = "./facetest/face_detector/opencv_face_detector.pbtxt";

    dnn::Net net = dnn::readNetFromTensorflow(pb_file_path,pbtxt_file_path);
    VideoCapture cap(0);
    Mat frame;
    //namedWindow("DNN人脸识别",WINDOW_FULLSCREEN);
    while(true){
        cap.read(frame);
        if(frame.empty()) break;
        Mat blob = dnn::blobFromImage(frame,1.0,Size(300,300),Scalar(104,177,123),false,false);
        net.setInput(blob);
        Mat probs = net.forward();
        Mat detectMat(probs.size[2],probs.size[3],CV_32F,probs.ptr<float>());
        for(int row = 0; row < detectMat.rows;row++){
            float conf = detectMat.at<float>(row,2);
            if(conf > 0.5){
                float x1 = detectMat.at<float>(row,3)*frame.cols;
                float y1 = detectMat.at<float>(row,4)*frame.rows;
                float x2 = detectMat.at<float>(row,5)*frame.cols;
                float y2 = detectMat.at<float>(row,6)*frame.rows;
                Rect box(x1,y1,x2-x1,y2-y1);
                rectangle(frame,box,Scalar(0,0,255),2,8);
            }
            imshow("DNN人脸识别",frame);
            char c = waitKey(1);
            if(c == 27) break;
        }
    }

    waitKey(0);
    destroyAllWindows();
    return 0;
    


}