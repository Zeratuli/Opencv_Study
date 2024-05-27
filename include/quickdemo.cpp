#include <quickopencv.h>


void QuickDemo::colorSpace_Demo(Mat &image,std::string name){
    Mat gray, hsv;
    cvtColor(image,hsv,COLOR_BGR2HSV);
    cvtColor(image,gray,COLOR_BGR2GRAY);
    namedWindow("HSV",WINDOW_FREERATIO);
    namedWindow("gray",WINDOW_FREERATIO);
    imshow("HSV",hsv);
    imshow("gray",gray);
    //imwrite("./image/" + name + "-hsv.jpeg",   hsv);
    //imwrite("./image/" + name + "-gray.jpeg", gray);
}

void QuickDemo::mat_creation_demo(Mat &image,std::string name){
    Mat m1,m2;
    m1 = image.clone();
    image.copyTo(m2);

    Mat m3 = Mat::ones(Size(8,8),CV_8UC3);
    m3 = Scalar(8,8,8);
    std::cout << m3.size() << " " << m3.cols << " " << m3.rows << std::endl;
    std::cout << m3 << std::endl;
    Mat m4 = (Mat_<char>(3,3) << 
                                0,-1,0,
                                100,5,100,
                                0,-1,0);
    namedWindow("my",WINDOW_FREERATIO);
    imshow("my",m4);
}

void QuickDemo::pixel_visit_demo(Mat &image){
    int h = image.rows;
    int w = image.cols;
    int dims = image.channels();
    for(int row = 0; row < h; row++){
        for(int col = 0; col < w; col++){
            if(dims == 1){
                int pv = image.at<uchar>(row,col);
                image.at<uchar>(row,col) = 255 - pv;
            }
            else if(dims == 3){
                Vec3b bgr = image.at<Vec3b>(row,col);
                image.at<Vec3b>(row,col)[0] = 255 - bgr[0];
                image.at<Vec3b>(row,col)[1] = 255 - bgr[1];
                image.at<Vec3b>(row,col)[2] = 255 - bgr[2];
            }
        }
    }
    imshow("show-test",image);
}

void QuickDemo::operators_demo(Mat &image){

    Mat now;
    Mat cheng = Mat::zeros(image.size(),image.type());
    Mat dst = Mat::zeros(image.size(),image.type());
    cheng = Scalar(20,20,20); 
    // int h = image.rows;
    // int w = image.cols;
    // int dims = image.channels();
    // for(int row = 0; row < h; row++){
    //     for(int col = 0; col < w; col++){
                // Vec3b p1 = image.at<Vec3b>(row,col);
                // Vec3b p2 = cheng.at<Vec3b>(row,col);
                // dst.at<Vec3b>(row,col)[0] = saturate_cast<uchar>(p1[0]+p2[0]);
                // dst.at<Vec3b>(row,col)[1] = saturate_cast<uchar>(p1[1]+p2[1]);
                // dst.at<Vec3b>(row,col)[2] = saturate_cast<uchar>(p1[2]+p2[2]);
    //     }
    // }

    multiply(image,cheng,now);//??????
    // add(image,cheng,now);
    // subtract(image,cheng,now);
    // divide(image,cheng,now);
    namedWindow("add-ope",WINDOW_FREERATIO);
    imshow("add-ope",now);
}

// static void on_lightness(int b,void *userdata){
//     const Mat image = *((Mat*)userdata);
//     Mat m = Mat::zeros(image.size(),image.type());
//     Mat dst = Mat::zeros(image.size(),image.type());
//     addWeighted(image,1.0,m,0,b,dst);
//     imshow("control",dst);
// }

// static void on_contrast(int b,void *userdata){
//     const Mat image = *((Mat*)userdata);
//     Mat m = Mat::zeros(image.size(),image.type());
//     Mat dst = Mat::zeros(image.size(),image.type());
//    // m = Scalar(b,b,b);
//     double contrast = b / 100.0;
//     addWeighted(image,contrast,m,0.0,0,dst);
//     imshow("control",dst);
// }

// void QuickDemo::track_bar_demo(Mat &image){
//     namedWindow("control",WINDOW_AUTOSIZE);
//     int max_lightness_value = 100;
//     int max_contrast_value = 200;
//     int lightness = 50;
//     int contrast_value = 100;
//     createTrackbar("Value-Bar","control",&lightness,max_lightness_value,on_lightness,(void*)(&image));
//     createTrackbar("Contrast-Bar","control",&contrast_value,200,on_contrast,(void*)(&image));
//     on_lightness(50,&image);
// }
static void on_lightness(int lightness, void* userdata) {
    // ??????????
    Mat image = *((Mat*)userdata);

    // ???????????????
    int contrast_value = getTrackbarPos("Contrast Bar:", "????????????");
    double contrast = contrast_value / 100.0;

    // ???????????
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());

    // ????????????
    addWeighted(image, contrast, m, 0, lightness, dst);

    // ??????
    imshow("????????????", dst);
}

static void on_contrast(int contrast_value, void* userdata) {
    // ??????????
    Mat image = *((Mat*)userdata);

    // ???????????????
    int lightness = getTrackbarPos("Value Bar:", "????????????");
    double contrast = contrast_value / 100.0;

    // ???????????
    Mat dst = Mat::zeros(image.size(), image.type());
    Mat m = Mat::zeros(image.size(), image.type());

    // ????????????
    addWeighted(image, contrast, m, 0.0, lightness, dst);

    // ??????
    imshow("????????????", dst);
}

void QuickDemo::track_bar_demo(Mat& image) {
    namedWindow("????????????", WINDOW_AUTOSIZE);

    // ??????????????
    int lightness = 50;
    int max_value = 100;
    int contrast_value = 100;

    // ???????????????????????
    createTrackbar("Value Bar:", "????????????", &lightness, max_value, on_lightness, (void*)(&image));
    createTrackbar("Contrast Bar:", "????????????", &contrast_value, 200, on_contrast, (void*)(&image));

    // ??????
    on_lightness(lightness, &image);
}

void QuickDemo::key_demo(Mat &image){
    Mat dst;
    while(true){
        int c = waitKey(100);
        if(c >= 49 and c <= 57){
            std::cout << "You entered key #" << c-48 << std::endl;
            switch(c){
                case 49:
                    cvtColor(image,dst,COLOR_BGR2GRAY);
                    break;
                case 50:
                    cvtColor(image,dst,COLOR_BGR2HSV);
                    break;
                case 51:
                    cvtColor(image,dst,COLOR_BGR2HSV);
                    break;
                case 52:
                    cvtColor(image,dst,COLOR_BGR2HLS);
                    break;
                case 53:
                    cvtColor(image,dst,COLOR_BGR2HLS_FULL);
                    break;
                case 54:
                    cvtColor(image,dst,COLOR_BGR2Luv);
                    break;
                case 55:
                    cvtColor(image,dst,COLOR_BGR2BGRA);
                    break;
                case 56:
                    dst = Scalar(50,50,50);
                    add(image,dst,dst);
                    break;
            }
            imshow("keyboard-bound",dst);
        }
        else 
            continue;
    }
}

void QuickDemo::color_style_demo(Mat &image){
    const int colorMap[21] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
    int temp = 0;
    Mat dst;    
    // while(1){
    //     int c = waitKey(100);
    //     applyColorMap(image,dst,colorMap[(temp++)%19]);
    //     //namedWindow("skd",WINDOW_AUTOSIZE);
    //     int matrix_size = 3000;
    //     Mat dst(matrix_size, matrix_size, CV_8UC3,Scalar(255, 255, 255)); // ??
    //     Mat image_mini; // ???????С?
    //     resize(dst, image_mini, Size(matrix_size/3, matrix_size/3)); // ??С????
    //     imshow("mini of original image", image_mini);

    //     //imshow("color-style",dst);
    // }
    while (1) {
        int c = waitKey(100);
        if (c == 27) break; // ?? ESC ????????

        //Mat dst;
        applyColorMap(image, dst, colorMap[(temp++) % 19]);

        Mat image_mini;
        resize(dst, image_mini, Size(dst.cols / 3, dst.rows / 3)); // ??С????
        imshow("color-style-image", image_mini);
    }
}

void QuickDemo::bitwise_demo(Mat &image){
    Mat m1 = Mat::zeros(Size(256,256),CV_8UC3);
    Mat m2 = Mat::zeros(Size(256,256),CV_8UC3);
    rectangle(m1,Rect(100,100,80,80),Scalar(255,255,0),-1,LINE_8,0);
    rectangle(m2,Rect(150,150,80,80),Scalar(0,255,255),-1,LINE_8,0);
    imshow("m1",m1);
    imshow("m2",m2);
    Mat dst;
    bitwise_not(image,dst);
    imshow("craft-operation1",dst);
    bitwise_xor(m1,m2,dst);
    imshow("craft-operation2",dst);
}

void QuickDemo::channels_demo(Mat &image){
    std::vector<Mat> mv;

    split(image,mv);
    Mat dst;
    mv[1] = 0,mv[2] = 0;
    merge(mv,dst);
    imshow("???",dst);

    split(image,mv);
    mv[0] = 0,mv[2] = 0;
    merge(mv,dst);
    imshow("???",dst);

    split(image,mv);
    mv[0] = 0,mv[1] = 0;
    merge(mv,dst);
    imshow("???",dst);

    split(image,mv);
    mv[2] = 0;//???????
    merge(mv,dst);
    imshow("????",dst);

    split(image,mv);
    mv[1] = 0;//???????
    merge(mv,dst);
    imshow("????",dst);

    split(image,mv);
    mv[0] = 0;//???????
    merge(mv,dst);
    imshow("????",dst);

    //int from_to[] = {0,2, 1,1, 2,0};
    int from_to[] = {0,2, 1,0, 2,1};
    mixChannels(&image,1,&dst,1,from_to,3);
    imshow("mix-channel",dst);
}

void QuickDemo::inrannge_demo(Mat &image){
    // Mat dst;
    // cvtColor(image,dst,COLOR_BGR2GRAY);
    // imshow("hsv",dst);
    // Mat mask;
    // inRange(dst,Scalar(0,0,0),Scalar(180,255,60),mask);
    // imshow("inrange",mask);

    // ????HSV??????
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // ????inRange????????°?????????????????
    cv::Mat mask1, mask2, mask;
    cv::inRange(hsv, cv::Scalar(0, 0, 0), cv::Scalar(180, 255, 50), mask1); // ?????Χ
    cv::inRange(hsv, cv::Scalar(0, 0, 200), cv::Scalar(180, 30, 255), mask2); // ?????Χ

    // ?????????????????
    cv::bitwise_or(mask1, mask2, mask);

    // ??????????????????????????С??
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);

    // ?????????С???????????
    cv::dilate(mask, mask, kernel, cv::Point(-1, -1), 2);
    cv::erode(mask, mask, kernel, cv::Point(-1, -1), 2);

    // ?????????°???
    cv::Mat zebra;
    cv::bitwise_and(image, image, zebra, mask);

    // ??????
    cv::imshow("?????", image);
    cv::imshow("????", mask);
    cv::imshow("????", zebra);
}

void QuickDemo::pixel_statistic_demo(Mat &image){
    std::vector<Mat> dst;
    split(image,dst);
    double maxv,minv;
    Point minLoc,maxLoc;
    for(int i = 0 ; i < dst.size(); i++){
        minMaxLoc(dst[i],&minv,&maxv,&minLoc,&maxLoc,Mat());
        std::cout << "NO." << i << " channels: " << "min value " << minv << " max value " << maxv <<std::endl;
    }
    
    Mat mean,stddev;
    Mat redback = Mat::zeros(image.size(),image.type());
    redback =  Scalar(40,40,200);
    meanStdDev(redback,mean,stddev);
    std::cout << "means: \n" << mean << std::endl;
    std::cout << "stddev: \n" << stddev << std::endl;
}

void QuickDemo::drawing_demo(Mat &image){
    Rect rect;
    rect.x = 200;
    rect.y = 200;
    rect.width = 100;
    rect.height = 100;

    Mat bg = Mat::zeros(image.size(),image.type());
    rectangle(bg, rect, Scalar(0,0,255),-1,8,0);
    circle(bg,Point(200,200),15,Scalar(255,0,0),2,LINE_AA,0);
    circle(bg,Point(300,200),15,Scalar(255,0,0),2,LINE_AA,0);
    circle(bg,Point(200,300),15,Scalar(255,0,0),2,LINE_AA,0);
    circle(bg,Point(300,300),15,Scalar(255,0,0),2,LINE_AA,0);
    line(bg,Point(200,200),Point(300,300),Scalar(0,255,0),2,8,0);
    line(bg,Point(300,200),Point(200,300),Scalar(0,255,0),2,8,0);
    RotatedRect rrt;
    rrt.center = Point(250,250);
    rrt.size = Size(100,200);
    rrt.angle = 180.0;
    ellipse(bg,rrt,Scalar(0,255,255),2,8);
    Mat dst;
    addWeighted(image,0.8,bg,0.2,0,dst);
    namedWindow("绘制演示",WINDOW_FREERATIO);
    imshow("绘制演示",dst);
}


void QuickDemo::random_drawing_demo(){
    Mat canvas = Mat::zeros(Size(512,512),CV_8UC3);
    int w = canvas.cols;
    int h = canvas.rows;
    RNG rng(12345);
    while(true){
        int c = waitKey(10);
        if(c == 27) break;
        int x1 = rng.uniform(0,w);
        int y1 = rng.uniform(0,h);
        int x2 = rng.uniform(0,w);
        int y2 = rng.uniform(0,h);
        int b = rng.uniform(0,255);
        int g = rng.uniform(0,255);
        int r = rng.uniform(0,255);
        line(canvas,Point(x1,y1),Point(x2,y2),Scalar(b,g,r),1,LINE_AA,0);
        imshow("canvas",canvas);
    }
}

void QuickDemo::polyline_drawing_demo(Mat &image){
    Mat canvas = Mat::zeros(Size(512,512),CV_8UC3);
    std::vector<Point> ve = {Point(100,100),Point(100,120),Point(200,300),Point(300,400),Point(120,120)};
    std::vector<std::vector<Point>> contour;
    contour.push_back(ve);
    // polylines(canvas,contour,true,Scalar(0,255,255),2,LINE_AA,0);
    // fillPoly(canvas,contour,Scalar(255,0,255),8,0);
    drawContours(canvas,contour,-2,Scalar(255,0,0),-1);
    imshow("polylines",canvas);

}
//point
Point sp(-1, -1);
Point ep(-1, -1);
Mat temp;
static void on_draw(int event, int x, int y, int flags, void *userdata) {
	Mat image = *((Mat*)userdata);
	if (event == EVENT_LBUTTONDOWN) {
		sp.x = x;
		sp.y = y;
		std::cout <<"start point:" << sp << std::endl;
	}
	else if (event == EVENT_LBUTTONUP) {
		ep.x = x;
		ep.y = y;
		int dx = ep.x - sp.x;
		int dy = ep.y - sp.y;
		if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				imshow("ROI区域", image(box));
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				imshow("鼠标绘制", image);
				// ready for next drawing
				sp.x = -1;
				sp.y = -1;
		}
	}
	else if (event == EVENT_MOUSEMOVE) {
		if (sp.x > 0 && sp.y > 0) {
			ep.x = x;
			ep.y = y;
			int dx = ep.x - sp.x;
			int dy = ep.y - sp.y;
			if (dx > 0 && dy > 0) {
				Rect box(sp.x, sp.y, dx, dy);
				temp.copyTo(image);
				rectangle(image, box, Scalar(0, 0, 255), 2, 8, 0);
				//circle(image, box, 15, Scalar(0, 0, 255), 2, 8, 0);
				imshow("鼠标绘制", image);
			}
		}
	}
}

// Point sp(-1, -1);  // 起始点
// Point ep(-1, -1);  // 结束点
// Mat temp;          // 用于保存原始图像

// static void on_draw(int event, int x, int y, int flags, void *userdata) {
//     Mat image = *((Mat*)userdata);  // 从用户数据中获取图像

//     // 鼠标左键按下事件
//     if (event == EVENT_LBUTTONDOWN) {
//         sp.x = x;  // 记录起始点 x 坐标
//         sp.y = y;  // 记录起始点 y 坐标
//         std::cout << "start point:" << sp << std::endl;
//     }
//     // 鼠标左键松开事件
//     else if (event == EVENT_LBUTTONUP) {
//         ep.x = x;  // 记录结束点 x 坐标
//         ep.y = y;  // 记录结束点 y 坐标
//         int dx = ep.x - sp.x;  // 计算水平距离
//         int dy = ep.y - sp.y;  // 计算垂直距离
//         int radius = sqrt(dx * dx + dy * dy);  // 计算半径

//         if (radius > 0) {
//             Point center(sp.x, sp.y);  // 定义圆心
//             temp.copyTo(image);  // 恢复原始图像
//             circle(image, center, radius, Scalar(0, 0, 255), 2, 8, 0);  // 在图像上绘制圆形
//             imshow("鼠标绘制", image);  // 显示带有圆形的图像

//             // 计算圆形的ROI区域并显示
//             Rect roi(center.x - radius, center.y - radius, 2 * radius, 2 * radius);
//             Mat roi_image = temp(roi);
//             Mat mask = Mat::zeros(roi_image.size(), roi_image.type());
//             circle(mask, Point(radius, radius), radius, Scalar(255, 255, 255), -1, 8, 0);
//             Mat roi_result;
//             roi_image.copyTo(roi_result, mask);
//             imshow("ROI区域", roi_result);

//             // 为下次绘制做好准备
//             sp.x = -1;
//             sp.y = -1;
//         }
//     }
//     // 鼠标移动事件
//     else if (event == EVENT_MOUSEMOVE) {
//         if (sp.x > 0 && sp.y > 0) {  // 如果起始点已被记录
//             ep.x = x;  // 更新结束点 x 坐标
//             ep.y = y;  // 更新结束点 y 坐标
//             int dx = ep.x - sp.x;  // 计算水平距离
//             int dy = ep.y - sp.y;  // 计算垂直距离
//             int radius = sqrt(dx * dx + dy * dy);  // 计算半径

//             if (radius > 0) {
//                 Point center(sp.x, sp.y);  // 定义圆心
//                 temp.copyTo(image);  // 恢复原始图像
//                 circle(image, center, radius, Scalar(0, 0, 255), 2, 8, 0);  // 在图像上绘制圆形
//                 imshow("鼠标绘制", image);  // 实时显示带有圆形的图像
//             }
//         }
//     }
// }

void QuickDemo::mouse_drawing_demo(Mat &image) {
	namedWindow("鼠标绘制", WINDOW_AUTOSIZE);
	setMouseCallback("鼠标绘制", on_draw, (void*)(&image));
	imshow("鼠标绘制", image);
	temp = image.clone();
}

void QuickDemo::norm_demo(Mat &image){
    Mat dst;
    std::cout << image.type() << std::endl;
    image.convertTo(image,CV_32F);
    normalize(image,dst,1.0,0,NORM_MINMAX);
    std::cout << dst.type() << std::endl;
    imshow("图像数据归一化",dst);
}

void QuickDemo::resize_demo(Mat &image){
    Mat zoomin,zoomout;
    int h = image.rows;
    int w = image.cols;
    resize(image,zoomin,Size(w,h),0,0,INTER_LINEAR);
    imshow("zoomin",zoomin);
}

void QuickDemo::flip_demo(Mat &image){
    Mat dst;
    flip(image,dst,0);
    imshow("translate",dst);
}

void QuickDemo::rotate_demo(Mat &image){
    Mat dst,M;
    int w = image.cols;
    int h = image.rows;
    M = getRotationMatrix2D(Point2f(w/2,h/2),90,1.0);
    double cos = abs(M.at<double>(0,0));
    double sin = abs(M.at<double>(0,1));
    int nw = cos*w + sin*h;
    int nh = sin*w + cos*h;
    M.at<double>(0,2) += (nw/2-w/2);
    M.at<double>(1,2) += (nh/2-h/2);
    warpAffine(image,dst,M,Size(nw,nh),INTER_LINEAR,0,Scalar(255,255,0));
    imshow("rotate-demo",dst);

}

void QuickDemo::video_demo(Mat &image){
    VideoCapture capture(0);//0就是开启摄像头
    int frame_width = capture.get(CAP_PROP_FRAME_WIDTH);
    int frame_height = capture.get(CAP_PROP_FRAME_HEIGHT);
    int count = capture.get(CAP_PROP_FRAME_COUNT);
    double fps = capture.get(CAP_PROP_FPS);
    //std::cout << "frame width" << frame_width << std::endl;
    //std::cout << "frame height" << frame_height << std::endl;
    //std::cout << "FPS:" << fps << std::endl;
    //std::cout << "Number of Frames:" << count << std::endl;
    //VideoWriter writer("C:/Users/54219/Desktop/config-opencv/video/New-5.mp4",capture.get(CAP_PROP_FOURCC),fps,Size(frame_width,frame_height),true);
    Mat frame;
    while(true){
        capture.read(frame);
        //flip(frame,frame,1);
        if(frame.empty()){
            break;
        }
        Mat dst;
        bilateralFilter(frame,dst,0,200,10);
        namedWindow("frame",WINDOW_FREERATIO);
        imshow("frame",dst);
        colorSpace_Demo(dst,"Arknights");
        int c = waitKey(30);
        if(c == 27) break;
    }
    capture.release();
}

void QuickDemo::showHistogram_demo(Mat &image){
    std::vector<Mat> bgr_plane;
    split(image,bgr_plane);
    const int channels[1] = {0};
    const int bins[1] = {256};
    float hranges[2] = {0,255};//高度范围
    const float *ranges[1] = {hranges};
    Mat b_hist;
    Mat g_hist;
    Mat r_hist;
    calcHist(&bgr_plane[0],1,0,Mat(),b_hist,1,bins,ranges);
    calcHist(&bgr_plane[1],1,0,Mat(),g_hist,1,bins,ranges);//绘制直方图
    calcHist(&bgr_plane[2],1,0,Mat(),r_hist,1,bins,ranges);
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w/bins[0]);
    Mat histImage = Mat::zeros(hist_h,hist_w,CV_8UC3);
    normalize(b_hist,b_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());//归一化
    normalize(g_hist,g_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
    normalize(r_hist,r_hist,0,histImage.rows,NORM_MINMAX,-1,Mat());
    for(int i = 1; i < bins[0]; i++){
        line(histImage,Point(bin_w*(i-1),hist_h-cvRound(b_hist.at<float>(i-1))),
            Point(bin_w*(i),hist_h-cvRound(b_hist.at<float>(i))),Scalar(255,0,0),2,8,0);
        line(histImage,Point(bin_w*(i-1),hist_h-cvRound(g_hist.at<float>(i-1))),//通过for循环不断绘制RGB分布
            Point(bin_w*(i),hist_h-cvRound(g_hist.at<float>(i))),Scalar(0,255,0),2,8,0);
        line(histImage,Point(bin_w*(i-1),hist_h-cvRound(r_hist.at<float>(i-1))),
            Point(bin_w*(i),hist_h-cvRound(r_hist.at<float>(i))),Scalar(0,0,255),2,8,0);
    }
    namedWindow("histogram Demo",WINDOW_AUTOSIZE);
    imshow("Histogram Demo",histImage);
}

void QuickDemo::histogram_2d_demo(Mat &image){
    Mat hsv,hs_hist;
    cvtColor(image,hsv,COLOR_BGR2HSV);
    int hbins = 30, sbins = 32;
    int hist_bins[] = {hbins,sbins};
    float h_range[] = {0,180};
    float s_range[] = {0,256};
    const float *hs_ranges[] = {h_range,s_range};
    int hs_channels[] = {0,1};
    calcHist(&hsv,1,hs_channels,Mat(),hs_hist,2,hist_bins,hs_ranges,true,false);
    double maxVal = 0;
    minMaxLoc(hs_hist,0,&maxVal,0,0);
    int scale = 10;
    Mat hist2d_image = Mat::zeros(sbins*scale,hbins*scale,CV_8UC3);
    for(int h = 0; h < hbins; h++){
        for(int s = 0; s < sbins; s++){
            float binVal = hs_hist.at<float>(h,s);
            int intensity = cvRound(binVal * 255/maxVal);
            rectangle(hist2d_image,Point(h*scale,s*scale),
                Point((h+1)*scale-1,(s+1)*scale-1),
                Scalar::all(intensity),
                -1);
        }
    }
    imshow("H-S Histogram",hist2d_image);
    imwrite("./image/hist_2d.jpg",hist2d_image);
}

void QuickDemo::histogram_eq_demo(Mat &image){
    Mat dst;
    cvtColor(image,dst,COLOR_BGR2GRAY);
    imshow("原灰度图像",dst);
    Mat now;
    equalizeHist(dst,now);
    imshow("均衡化后：",now);
}

void QuickDemo::linear_convolution_demo(Mat &image){
    Mat dst;
    blur(image,dst,Size(23,23),Point(-1,-1));
    imshow("图像模糊",dst);
}

void QuickDemo::gaussian_blur_demo(Mat &image){
    Mat dst;
    GaussianBlur(image,dst,Size(5,5),15);
    imshow("高斯图像模糊",dst);
}

void QuickDemo::bifilter_demo(Mat &image){
    Mat dst;
    bilateralFilter(image,dst,0,100,10);
    imshow("双边模糊",dst);
}

