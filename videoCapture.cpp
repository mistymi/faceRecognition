#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

string haar_face_datapath = "D:/opencv/opencv/build/etc/haarcascades/haarcascade_frontalface_alt_tree.xml";
int main(int argc, char** argv) {

	// 打开摄像头
	VideoCapture capture(0); 
	if (!capture.isOpened()) {
		printf("could not open camera...\n");
		return -1;
	}

	Size S = Size((int)capture.get(CV_CAP_PROP_FRAME_WIDTH),(int)capture.get(CV_CAP_PROP_FRAME_HEIGHT));//获得摄像头捕获数据宽度高度
	int fps = capture.get(CV_CAP_PROP_FPS);//得到文件的fps
	CascadeClassifier faceDetector;//检测是否有人脸
	faceDetector.load(haar_face_datapath);
	Mat frame;
	namedWindow("camera_capture", CV_WINDOW_AUTOSIZE);//显示视屏内容

	//捕获视频内容
	vector<Rect> faces;
	int count = 0;
	while (capture.read(frame))
	{
		flip(frame, frame, 1);//旋转视频
		faceDetector.detectMultiScale(frame, faces, 1.1, 1, 0, Size(100, 120), Size(380, 400));
		for (int i = 0; i < faces.size(); i++)
		{
			if (count % 10 == 0)
			{
				Mat dst;
				resize(frame(faces[i]), dst, Size(120, 110));//设置人脸大小
				imwrite(format("C:/Users/Administrator/Desktop/imageoutput/face_%d.jpg", count), dst);//把捕获的图片存到指定文件夹
			}
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
		}
		imshow("carmera_capture", frame);//将图像在特定窗口进行显示
		char c = waitKey(10);
		if (c == 27)
			break;
		count++;
	}
	waitKey(0);
	return 0;
}
