#include "opencv2/opencv.hpp"
#include <iostream>
#include <vector>
 
using namespace std;
using namespace cv;
 
int my_mean_shift(Mat src_img, Rect& roi_window, int max_iter_times, double threshold)
{	
	//判断图片是不是uint8，否则之前处理可能出错了
	if(src_img.type() != CV_8UC1)
	{
		return -1;
	}

	//在原始图片上初始化兴趣区域（矩型），并确定中心
	Mat roi_img = src_img(roi_window);
	int x_c = roi_img.cols / 2;
	int y_c = roi_img.rows / 2;
	int width = src_img.cols*src_img.cols + src_img.rows*src_img.rows;
	
	//开始meanshift主体，循环次数为最大迭代次数
	for(int i = 0; i < max_iter_times; i++)
	{
		float x_dst=0.0, y_dst=0.0;
		float weight_sum = 0;
		/* 计算meanshift向量（重心相对于中心的偏移） */
		for(int y_axis = 0; y_axis < roi_img.rows; y_axis++)
		{
			for(int x_axis = 0; x_axis< roi_img.cols; x_axis++)
			{	
				//如果有值就计算其对整体重心的影响
				if(roi_img.at<unsigned char>(y_axis, x_axis))
				{
					int weight = roi_img.at<unsigned char>(y_axis, x_axis);
					//核函数使用核函数 1 - x*x
					weight = weight*(1-((x_axis - x_c)*(x_axis - x_c) + (y_axis - y_c)*(y_axis - y_c))/width);
					weight_sum += weight;
					x_dst += (x_axis - x_c)*weight;
					y_dst += (y_axis - y_c)*weight;
				}
			}
		}

		//进行归一化后移动兴趣区域
		x_dst /= weight_sum;
		y_dst /= weight_sum;
		roi_window.x += x_dst;
		roi_window.y += y_dst;
		//如果偏移小于一定值就不再移动
		if(fabs(x_dst) + fabs(y_dst) < threshold)
			break;
		//更新兴趣区域，再次计算偏移
		roi_img = src_img(roi_window);
	}
	return 0;
}


void my_split(const Mat& src, Mat& dst){
	int width = src.cols;
	int height = src.rows;
	int len = src.rows * src.cols;

	uchar* pSrc = (uchar*)src.data;
	uchar* pDst = (uchar*)dst.data;
	
	//读取第二个S通道的数据
	for (int j = 0; j < len; j++) {
		pDst[j] = pSrc[j * 3 + 1];
	}
}


int main()
{
	//读取两帧图像
	Mat image_1 = imread("1.jpg");
	Mat image_2 = imread("2.jpg");
	Mat imageROI = image_1(Rect(178, 197, 25, 85));
	Mat image_display;
	image_1.copyTo(image_display);
	rectangle(image_display, Rect(178, 197, 25, 85), Scalar(0, 0, 0));
	imshow("origin_image_hsv", image_display);
 
	Mat image_hsv;
	//为了更好的数字化图像将BGR转为HSV格式
	cvtColor(imageROI, image_hsv, CV_BGR2HSV);
	Mat mask;
	int lower_bound = 125;
	//对饱和度通道进行二值化处理，减少数据量，凸显出目标的轮廓
	Mat v(Size(image_hsv.cols,image_hsv.rows), CV_8UC1);
	//主要用颜色信息，只读取S通道数据
	my_split(image_hsv, v);
	//通过二值化制作mask，加速计算，同时也去除干扰
	threshold(v, mask, lower_bound, 255, THRESH_BINARY);

	//设置计算直方图需要的参数
	int histSize[1];
	const float* ranges[1];
	int channels[1];
	histSize[0] = 256;
	float tmp[2] = {0.0, 256.0};
	ranges[0] = tmp;
	channels[0] = 0;
	MatND hist_hsv;
	//根据参数计算直方图，保存到hist_hsv中
	calcHist(&image_hsv, 1, channels, mask, hist_hsv, 1, histSize, ranges);
	normalize(hist_hsv, hist_hsv, 1.0);
	
	//与处理兴趣区域一样处理待检测的图像image_2
	Mat image_2_hsv;
	cvtColor(image_2, image_2_hsv, CV_BGR2HSV);
	Mat v_2(Size(image_2_hsv.cols,image_2_hsv.rows), CV_8UC1);
	my_split(image_2_hsv, v_2);
	threshold(v_2, v_2, lower_bound, 255, THRESH_BINARY);
 
	Mat processed_hsv;
	//计算反投影，将选取兴趣区域的直方图hist_hsv反向投影到时间序列靠后的图像image_2_hsv上
	calcBackProject(&image_2_hsv, 1, channels, hist_hsv, processed_hsv, ranges, 255.0);
	bitwise_and(processed_hsv, v_2, processed_hsv);
	imshow("processed_hsv", processed_hsv);
 
	//设置初始兴趣区域，计算时序靠后图像中的目标位置
	Rect rect2(180, 200, 35, 95);
	my_mean_shift(processed_hsv, rect2, 300, 0.01);
	rectangle(image_2, rect2, Scalar(0, 0, 0));
	imshow("image_res", image_2);
 
	cv::waitKey();
	return 0;
 
}
 
 
