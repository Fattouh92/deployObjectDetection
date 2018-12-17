#include <iostream>
#include "ObjectDetection.h"

int main() {

    // TODO: provide your working directory where input image and frozen graph residegraph
    string root_dir = "/home/mohamed/code/deployObjectDetection/";
    string image = "data/grace_hopper.jpg";
    string graph = "data/inception_v3_2016_08_28_frozen.pb";

    // TODO: provide your own model configurations
    int32 input_width = 299;
    int32 input_height = 299;
    string input_layer = "input";
    string output_layer = "InceptionV3/Predictions/Reshape_1";

    Mat inputImg = imread(root_dir+image);
    ObjectDetectionWrapper predictor(input_width, input_height, input_layer, output_layer);

    // load frozen graph
    Status load_graph_status = predictor.load_graph(root_dir+graph);
    if (!load_graph_status.ok()) {
        LOG(ERROR) << load_graph_status;
        return -1;
    }

    // forward path on model
    vector<Tensor> outputs = predictor.forward_path(inputImg);
    if(outputs.empty()) {
        LOG(ERROR) << "Running model failed";
        return -1;
    }

    // TODO: post-processing of network output

    return 0;
}