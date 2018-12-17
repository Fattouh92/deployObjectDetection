#include "ObjectDetection.h"

ObjectDetectionWrapper::ObjectDetectionWrapper(int32 input_width_, int32 input_height_, const string& input_tensor_name_, const string& output_tensor_name_) {
    input_width =input_width_;
    input_height=input_height_;
    input_tensor_name=input_tensor_name_;
    output_tensor_name=output_tensor_name_;
}

Status ObjectDetectionWrapper::load_graph(string graph_path) {
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_path, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_path, "'");
    }

    session.reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = session->Create(graph_def);

    return session_create_status;
}

Tensor ObjectDetectionWrapper::readTensorFromMat(Mat &mat) {
    int depth = mat.channels();
    int batch = 1;

    Tensor inputTensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch, input_height, input_width, depth}));
    auto inputTensorMapped = inputTensor.tensor<float, 4>();

    const tensorflow::uint8 *source_data = (tensorflow::uint8 *) mat.data;

    const tensorflow::uint8 *all_sources[1] = {source_data};

    for (int b=0; b<batch; b++){
        const tensorflow::uint8 *source_data_temp = all_sources[b];
        for (int y = 0; y < input_height; y++) {
            const tensorflow::uint8 *source_row = source_data_temp + (y * input_width * depth);
            for (int x = 0; x < input_width; x++) {
                const tensorflow::uint8 *source_pixel = source_row + (x * depth);

                const tensorflow::uint8 *source_value_blue = source_pixel;
                const tensorflow::uint8 *source_value_green = source_pixel + 1;
                const tensorflow::uint8 *source_value_red = source_pixel + 2;

                inputTensorMapped(b, y, x, 0) = (*source_value_red) / 255.;
                inputTensorMapped(b, y, x, 1) = (*source_value_green) / 255.;
                inputTensorMapped(b, y, x, 2) = (*source_value_blue) / 255.;
            }
        }
    }

    return inputTensor;
}

vector<Tensor> ObjectDetectionWrapper::forward_path(Mat camera_image) {
    Mat input_mat;
    vector<Tensor> outputs;
    Tensor inputTensor;

    resize(camera_image, input_mat, Size(input_width, input_height));
    inputTensor = ObjectDetectionWrapper::readTensorFromMat(input_mat);

    Status run_status = session->Run({{input_tensor_name, inputTensor}},
                                        {output_tensor_name}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
    }

    return outputs;
}
