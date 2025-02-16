#include <opencv2/opencv.hpp>
#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#include <iostream>

using namespace cv;
using namespace tensorflow;
using namespace std;

// Function to preprocess the image
Tensor preprocessImage(const string &imagePath) {
    Mat img = imread(imagePath, IMREAD_GRAYSCALE);
    if (img.empty()) {
        cerr << "Error: Could not load image!" << endl;
        exit(1);
    }

    resize(img, img, Size(28, 28));
    img.convertTo(img, CV_32F, 1.0 / 255);
    Tensor inputTensor(DT_FLOAT, TensorShape({1, 28, 28, 1}));
    auto inputMapped = inputTensor.tensor<float, 4>();

    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            inputMapped(0, i, j, 0) = img.at<float>(i, j);
        }
    }

    return inputTensor;
}

// Load and run model
void recognizeDigit(const string &modelPath, const string &imagePath) {
    Session* session;
    Status status = NewSession(SessionOptions(), &session);

    GraphDef graphDef;
    status = ReadBinaryProto(Env::Default(), modelPath, &graphDef);
    session->Create(graphDef);

    Tensor inputTensor = preprocessImage(imagePath);
    vector<pair<string, Tensor>> inputs = {{"conv2d_input", inputTensor}};
    vector<Tensor> outputs;

    status = session->Run(inputs, {"dense_1/Softmax"}, {}, &outputs);

    auto outputMapped = outputs[0].flat<float>();
    int predictedDigit = distance(outputMapped.data(), max_element(outputMapped.data(), outputMapped.data() + 10));

    cout << "Predicted Digit: " << predictedDigit << endl;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <model.pb> <image.png>" << endl;
        return 1;
    }

    string modelPath = argv[1];
    string imagePath = argv[2];

    recognizeDigit(modelPath, imagePath);
    return 0;
}
