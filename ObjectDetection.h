#ifndef WRAPPER_H
#define WRAPPER_H

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/opencv.hpp>

using namespace tensorflow;
using namespace cv;
using namespace std;

class ObjectDetectionWrapper {
    int32 input_width;
    int32 input_height;
    string input_tensor_name;
    string output_tensor_name;
    unique_ptr<Session> session;
    GraphDef graph_def;

public:
    ObjectDetectionWrapper(int32 input_width_, int32 input_height_, const string& input_tensor_name_, const string& output_tensor_name_);

    Status load_graph(string);
    vector<Tensor> forward_path(Mat);
    Tensor readTensorFromMat(Mat &mat);
};



#endif //WRAPPER_H
