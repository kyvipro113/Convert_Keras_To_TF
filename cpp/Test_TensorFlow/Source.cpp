#include<iostream>

#include<tensorflow/c/c_api.h>
#include<tensorflow/core/framework/graph.pb.h>
#include<tensorflow/core/public/session.h>
#include<tensorflow/core/public/version.h>
#include <tensorflow/core/platform/env.h>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc.hpp>

using namespace std;
using namespace tensorflow;

int main()
{
	cout << "Version Tensorflow: " << TF_Version() << endl;

	// Create session
	unique_ptr<Session> session(NewSession({}));

	// Create graph and read .pb file
	GraphDef graph_def;
	Status status = ReadBinaryProto(Env::Default(), "Model/unet_brain_segmentation.pb", &graph_def);
	cout << "Read file .pb:" << status.ToString() << endl;

	// Add graph to the session
	status = session->Create(graph_def);
	cout << "Add graph to session: " << status.ToString() << endl;

	// Preprocessing image and convert image to tensor
	const int HEIGHT = 256;
	const int WIDTH = 256;
	const int CHANELS = 3;

	cv::Mat image = cv::imread("data/image_test/TCGA_CS_6186_20000601_16.tif");
	cv::imshow("Image", image);
	cout << image.rows << " " << image.cols << endl;

	// Resize image
	cv::resize(image, image, cv::Size(HEIGHT, WIDTH));

	// Convert BGR image to RGB image
	cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

	// Convert Uint8 to Float32
	cv::Mat imgNorm;
	image.convertTo(imgNorm, CV_32FC3, 1.f / 255);

	cv::waitKey();

	// Convert image to tensor
	Tensor inputImg(DT_FLOAT, TensorShape({ 1, HEIGHT, WIDTH, CHANELS }));
	vector<Tensor> outputs;

	auto tensor_map = inputImg.tensor<float, 4>();

	for (int h = 0; h < HEIGHT; h++)
	{
		const float* source_row = ((float*)imgNorm.data) + (h * WIDTH * CHANELS);
		for (int w = 0; w < WIDTH; w++)
		{
			const float* source_pixel = source_row + (w * CHANELS);
			tensor_map(0, h, w, 0) = source_pixel[2];
			tensor_map(0, h, w, 1) = source_pixel[1];
			tensor_map(0, h, w, 2) = source_pixel[0];
		}
	}

	cout << inputImg.shape() << endl;

	// Inference model
	status = session->Run({ { "x", inputImg } }, { "Identity" }, {}, &outputs);
	cout << "Inferential state: " << status.ToString() << endl;

	// Convert predict tensor to image 
	cout << outputs[0].DebugString() << endl;

	Tensor result = outputs[0];
	cv::Mat imgPred(outputs[0].dim_size(1), outputs[0].dim_size(2), CV_32FC1, outputs[0].flat<float>().data());

	//imgPred = imgPred * 255;
	cv::normalize(imgPred, imgPred, 0, 255, cv::NORM_MINMAX, CV_8U);

	cv::imshow("Mask predict", imgPred);
	cv::waitKey();

	//cout << imgPred << endl;

	system("pause");
	return 0;
}
