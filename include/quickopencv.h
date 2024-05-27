#include <opencv2/opencv.hpp>
#include <string>
using namespace cv;

class QuickDemo{
    public:
        void colorSpace_Demo(Mat &image,std::string name);
        void mat_creation_demo(Mat &image,std::string name);
        void pixel_visit_demo(Mat &image);
        void operators_demo(Mat &image);
        void track_bar_demo(Mat &image);
        void key_demo(Mat &image);
        void color_style_demo(Mat &image);
        void bitwise_demo(Mat &image);
        void channels_demo(Mat &image);
        void inrannge_demo(Mat &image);
        void pixel_statistic_demo(Mat &image);
        void drawing_demo(Mat &image);
        void random_drawing_demo();
        void polyline_drawing_demo(Mat &image);
        void mouse_drawing_demo(Mat &image);
        void norm_demo(Mat &image);
        void resize_demo(Mat &image);
        void flip_demo(Mat &image);
        void rotate_demo(Mat &image);
        void video_demo(Mat &image);
        void showHistogram_demo(Mat &image);
        void histogram_2d_demo(Mat &image);
        void histogram_eq_demo(Mat &image);
        void linear_convolution_demo(Mat &image);
        void gaussian_blur_demo(Mat &image);
        void bifilter_demo(Mat &image);
};
