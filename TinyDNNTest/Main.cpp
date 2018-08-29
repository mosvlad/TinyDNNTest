#include <iostream>

#include "tiny_dnn/tiny_dnn.h"

int main() {
	// create a simple network with 2 layer of 10 neurons each
	// input is x, output is sin(x)
	tiny_dnn::network<tiny_dnn::sequential> net;
	net << tiny_dnn::fully_connected_layer(1, 10);
	net << tiny_dnn::relu_layer();
	net << tiny_dnn::fully_connected_layer(10, 50);
	net << tiny_dnn::relu_layer();
	net << tiny_dnn::fully_connected_layer(50, 50);
	net << tiny_dnn::relu_layer();
	net << tiny_dnn::fully_connected_layer(50, 5);
	net << tiny_dnn::sigmoid_layer();
	net << tiny_dnn::fully_connected_layer(5, 1);

	// create input and desired output on a period
	//std::vector<tiny_dnn::vec_t> X;
	//std::vector<tiny_dnn::vec_t> sinusX;
	//for (float x = -3.1416f; x < 3.1416f; x += 0.1f) {
	//	tiny_dnn::vec_t vx = { x };
	//	tiny_dnn::vec_t vsinx = { sinf(x) };

	//	X.push_back(vx);
	//	sinusX.push_back(vsinx);
	//}

	std::vector<tiny_dnn::vec_t> X;
	std::vector<tiny_dnn::vec_t> sinusX;
	for (float x = 0; x < 10.0; x += 0.5) {
		tiny_dnn::vec_t vx = { x };
		tiny_dnn::vec_t vsinx = { x * x };

		X.push_back(vx);
		sinusX.push_back(vsinx);
	}

	// set learning parameters
	size_t batch_size = 16;    // 16 samples for each network weight update
	int epochs = 10000;  // 10000 presentation of all samples
	tiny_dnn::adam opt;

	// this lambda function will be called after each epoch
	int iEpoch = 0;
	auto on_enumerate_epoch = [&]() {
		iEpoch++;
		if (iEpoch % 100) return;

		double loss = net.get_loss<tiny_dnn::mse>(X, sinusX);
		std::cout << "epoch=" << iEpoch << "/" << epochs << " loss=" << loss
			<< std::endl;
	};

	// learn
	std::cout << "learning the sinus function with 10000 epochs:" << std::endl;
	net.fit<tiny_dnn::mse>(opt, X, sinusX, batch_size, epochs, []() {},
		on_enumerate_epoch);

	std::cout << std::endl
		<< "Training finished, now computing prediction results:"
		<< std::endl;

	// compare prediction and desired output
	//float fMaxError = 0.f;
	//for (float x = -3.1416f; x < 3.1416f; x += 0.2f) {
	//	tiny_dnn::vec_t xv = { x };
	//	float fPredicted = net.predict(xv)[0];
	//	float fDesired = sinf(x);

	//	std::cout << "x=" << x << " sinX=" << fDesired
	//		<< " predicted=" << fPredicted << std::endl;

	//	// update max error
	//	float fError = fabs(fPredicted - fDesired);

	//	if (fMaxError < fError) fMaxError = fError;
	//}

	//std::cout << std::endl << "max_error=" << fMaxError << std::endl;
	//std::cin.get();

	while (true)
	{
		float x;
		std::cout << "x = ";
		std::cin >> x;
		
		tiny_dnn::vec_t xv = { x };
		float fPredicted = net.predict(xv)[0];
		float fDesired = x * x;

		std::cout << "predicted x^2: \t\t|" << fPredicted << std::endl;
		//std::cout << "sin(x): \t\t|" << fDesired << std::endl;
		std::cout << "x^2: \t\t\t|" << fDesired << std::endl;
		std::cout << "loss: \t\t\t|" << fDesired - fPredicted << std::endl;
		std::cout << std::endl;
		std::cin.get();
	}

	return 0;
}