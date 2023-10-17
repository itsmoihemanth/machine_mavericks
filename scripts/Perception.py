import tensorflow as tf
from tensorflow.python.tools import strip_unused_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import gfile

def load_saved_model(path):
    the_graph = tf.Graph()
    with tf.Session(graph=the_graph) as sess:
        tf.saved_model.loader.load(sess,
                [tf.saved_model.tag_constants.SERVING], path)
    return the_graph

saved_model_path = "ssdlite_mobilenet_v2_coco_2018_05_09/saved_model"
the_graph = load_saved_model(saved_model_path)

frozen_model_file = "frozen_model.pb"
input_node = "Preprocessor/sub"
bbox_output_node = "concat"
class_output_node = "Postprocessor/convert_scores"

def optimize_graph(graph):
    gdef = strip_unused_lib.strip_unused(
            input_graph_def = graph.as_graph_def(),
            input_node_names = [input_node],
            output_node_names = [bbox_output_node, class_output_node],
            placeholder_type_enum = dtypes.float32.as_datatype_enum)

    with gfile.GFile(frozen_model_file, "wb") as f:
        f.write(gdef.SerializeToString())

optimize_graph(the_graph)

import tfcoreml

coreml_model_path = "MobileNetV2_SSDLite.mlmodel"

input_width = 300
input_height = 300

input_tensor = input_node + ":0"
bbox_output_tensor = bbox_output_node + ":0"
class_output_tensor = class_output_node + ":0"

ssd_model = tfcoreml.convert(
    tf_model_path=frozen_model_file,
    mlmodel_path=coreml_model_path,
    input_name_shape_dict={ input_tensor: [1, input_height, input_width, 3] },
    image_input_names=input_tensor,
    output_feature_names=[bbox_output_tensor, class_output_tensor],
    is_bgr=False,
    red_bias=-1.0,
    green_bias=-1.0,
    blue_bias=-1.0,
    image_scale=2./255)

spec = ssd_model.get_spec()

spec.description.input[0].name = "image"
spec.description.input[0].shortDescription = "Input image"
spec.description.output[0].name = "scores"
spec.description.output[0].shortDescription = "Predicted class scores for each bounding box"
spec.description.output[1].name = "boxes"
spec.description.output[1].shortDescription = "Predicted coordinates for each bounding box"

input_mlmodel = input_tensor.replace(":", "__").replace("/", "__")
class_output_mlmodel = class_output_tensor.replace(":", "__").replace("/", "__")
bbox_output_mlmodel = bbox_output_tensor.replace(":", "__").replace("/", "__")

for i in range(len(spec.neuralNetwork.layers)):
    if spec.neuralNetwork.layers[i].input[0] == input_mlmodel:
        spec.neuralNetwork.layers[i].input[0] = "image"
    if spec.neuralNetwork.layers[i].output[0] == class_output_mlmodel:
        spec.neuralNetwork.layers[i].output[0] = "scores"
    if spec.neuralNetwork.layers[i].output[0] == bbox_output_mlmodel:
        spec.neuralNetwork.layers[i].output[0] = "boxes"

spec.neuralNetwork.preprocessing[0].featureName = "image"
  type
   {
    multiArrayType
    {
      dataType: DOUBLE
    }
    }
num_classes = 90
num_anchors = 1917
spec.description.output[0].type.multiArrayType.shape.append(num_classes + 1)
spec.description.output[0].type.multiArrayType.shape.append(num_anchors)

multiArrayType
 {
    shape: 4
    shape: 1917
    shape: 1
    dataType: DOUBLE
  }
del spec.description.output[1].type.multiArrayType.shape[-1]
import coremltools
spec = coremltools.utils.convert_neural_network_spec_weights_to_fp16(spec)
ssd_model = coremltools.models.MLModel(spec)
ssd_model.save(coreml_model_path)

import numpy as np

def get_anchors(graph, tensor_name):
    image_tensor = graph.get_tensor_by_name("image_tensor:0")
    box_corners_tensor = graph.get_tensor_by_name(tensor_name)
    box_corners = sess.run(box_corners_tensor, feed_dict={
        image_tensor: np.zeros((1, input_height, input_width, 3))})

    ymin, xmin, ymax, xmax = np.transpose(box_corners)
    width = xmax - xmin
    height = ymax - ymin
    ycenter = ymin + height / 2.
    xcenter = xmin + width / 2.
    return np.stack([ycenter, xcenter, height, width])

anchors_tensor = "Concatenate/concat:0"
with the_graph.as_default():
    with tf.Session(graph=the_graph) as sess:
        anchors = get_anchors(the_graph, anchors_tensor)

for b in 0..<numAnchors {
  // Read the anchor coordinates:
  let ay = anchors[b               ]
  let ax = anchors[b + numAnchors  ]
  let ah = anchors[b + numAnchors*2]
  let aw = anchors[b + numAnchors*3]

  // Read the predicted coordinates:
  let ty = boxes[b               ]
  let tx = boxes[b + numAnchors  ]
  let th = boxes[b + numAnchors*2]
  let tw = boxes[b + numAnchors*3]

  // The decoding formula:
  let x = (tx / 10) * aw + ax
  let y = (ty / 10) * ah + ay
  let w = exp(tw / 5) * aw
  let h = exp(th / 5) * ah

  // The final bounding box is given by (x, y, w, h)
}

from coremltools.models import datatypes
from coremltools.models import neural_network

input_features = [
    ("scores", datatypes.Array(num_classes + 1, num_anchors, 1)),
    ("boxes", datatypes.Array(4, num_anchors, 1))
]

output_features = [
    ("raw_confidence", datatypes.Array(num_anchors, num_classes)),
    ("raw_coordinates", datatypes.Array(num_anchors, 4))
]

builder = neural_network.NeuralNetworkBuilder(input_features, output_features)

builder.add_permute(name="permute_scores",
                    dim=(0, 3, 2, 1),
                    input_name="scores",
                    output_name="permute_scores_output")
builder.add_slice(name="slice_scores",
                  input_name="permute_scores_output",
                  output_name="raw_confidence",
                  axis="width",
                  start_index=1,
                  end_index=num_classes + 1)
x = (tx / 10) * aw + ax
y = (ty / 10) * ah + ay
w = exp(tw / 5) * aw
h = exp(th / 5) * ah

builder.add_slice(name="slice_yx",
                  input_name="boxes",
                  output_name="slice_yx_output",
                  axis="channel",
                  start_index=0,
                  end_index=2)

builder.add_elementwise(name="scale_yx",
                        input_names="slice_yx_output",
                        output_name="scale_yx_output",
                        mode="MULTIPLY",
                        alpha=0.1)

anchors_yx = np.expand_dims(anchors[:2, :], axis=-1)
anchors_hw = np.expand_dims(anchors[2:, :], axis=-1)

builder.add_load_constant(name="anchors_yx",
                          output_name="anchors_yx",
                          constant_value=anchors_yx,
                          shape=[2, num_anchors, 1])

builder.add_load_constant(name="anchors_hw",
                          output_name="anchors_hw",
                          constant_value=anchors_hw,
                          shape=[2, num_anchors, 1])
builder.add_elementwise(name="yw_times_hw",
                        input_names=["scale_yx_output", "anchors_hw"],
                        output_name="yw_times_hw_output",
                        mode="MULTIPLY")
builder.add_elementwise(name="decoded_yx",
                        input_names=["yw_times_hw_output", "anchors_yx"],
                        output_name="decoded_yx_output",
                        mode="ADD")

builder.add_slice(name="slice_hw",
                  input_name="boxes",
                  output_name="slice_hw_output",
                  axis="channel",
                  start_index=2,
                  end_index=4)

builder.add_elementwise(name="scale_hw",
                        input_names="slice_hw_output",
                        output_name="scale_hw_output",
                        mode="MULTIPLY",
                        alpha=0.2)

builder.add_unary(name="exp_hw",
                  input_name="scale_hw_output",
                  output_name="exp_hw_output",
                  mode="exp")

builder.add_elementwise(name="decoded_hw",
                        input_names=["exp_hw_output", "anchors_hw"],
                        output_name="decoded_hw_output",
                        mode="MULTIPLY")

builder.add_slice(name="slice_y",
                  input_name="decoded_yx_output",
                  output_name="slice_y_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

builder.add_slice(name="slice_x",
                  input_name="decoded_yx_output",
                  output_name="slice_x_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

builder.add_slice(name="slice_h",
                  input_name="decoded_hw_output",
                  output_name="slice_h_output",
                  axis="channel",
                  start_index=0,
                  end_index=1)

builder.add_slice(name="slice_w",
                  input_name="decoded_hw_output",
                  output_name="slice_w_output",
                  axis="channel",
                  start_index=1,
                  end_index=2)

builder.add_elementwise(name="concat",
                        input_names=["slice_x_output", "slice_y_output",
                                     "slice_w_output", "slice_h_output"],
                        output_name="concat_output",
                        mode="CONCAT")

builder.add_permute(name="permute_output",
                    dim=(0, 3, 2, 1),
                    input_name="concat_output",
                    output_name="raw_coordinates")
decoder_model = coremltools.models.MLModel(builder.spec)
decoder_model.save("Decoder.mlmodel")

nms_spec = coremltools.proto.Model_pb2.Model()
nms_spec.specificationVersion = 3

for i in range(2):
    decoder_output = decoder_model._spec.description.output[i].SerializeToString()

    nms_spec.description.input.add()
    nms_spec.description.input[i].ParseFromString(decoder_output)

    nms_spec.description.output.add()
    nms_spec.description.output[i].ParseFromString(decoder_output)

nms_spec.description.output[0].name = "confidence"
nms_spec.description.output[1].name = "coordinates"

output_sizes = [num_classes, 4]
for i in range(2):
    ma_type = nms_spec.description.output[i].type.multiArrayType
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[0].lowerBound = 0
    ma_type.shapeRange.sizeRanges[0].upperBound = -1
    ma_type.shapeRange.sizeRanges.add()
    ma_type.shapeRange.sizeRanges[1].lowerBound = output_sizes[i]
    ma_type.shapeRange.sizeRanges[1].upperBound = output_sizes[i]
    del ma_type.shape[:]

nms = nms_spec.nonMaximumSuppression
nms.confidenceInputFeatureName = "raw_confidence"
nms.coordinatesInputFeatureName = "raw_coordinates"
nms.confidenceOutputFeatureName = "confidence"
nms.coordinatesOutputFeatureName = "coordinates"
nms.iouThresholdInputFeatureName = "iouThreshold"
nms.confidenceThresholdInputFeatureName = "confidenceThreshold"

default_iou_threshold = 0.6
default_confidence_threshold = 0.4
nms.iouThreshold = default_iou_threshold
nms.confidenceThreshold = default_confidence_threshold
nms.pickTop.perClass = True
labels = np.loadtxt("coco_labels.txt", dtype=str, delimiter="\n")
nms.stringClassLabels.vector.extend(labels)
nms_model = coremltools.models.MLModel(nms_spec)
nms_model.save("NMS.mlmodel")

from coremltools.models.pipeline import *

input_features = [ ("image", datatypes.Array(3, 300, 300)),
                   ("iouThreshold", datatypes.Double()),
                   ("confidenceThreshold", datatypes.Double()) ]

output_features = [ "confidence", "coordinates" ]

pipeline = Pipeline(input_features, output_features)
ssd_output = ssd_model._spec.description.output
ssd_output[0].type.multiArrayType.shape[:] = [num_classes + 1, num_anchors, 1]
ssd_output[1].type.multiArrayType.shape[:] = [4, num_anchors, 1]
pipeline.add_model(ssd_model)
pipeline.add_model(decoder_model)
pipeline.add_model(nms_model)
pipeline.spec.description.input[0].ParseFromString(
    ssd_model._spec.description.input[0].SerializeToString())
pipeline.spec.description.output[0].ParseFromString(
    nms_model._spec.description.output[0].SerializeToString())
pipeline.spec.description.output[1].ParseFromString(
    nms_model._spec.description.output[1].SerializeToString())

pipeline.spec.description.input[1].shortDescription = "(optional) IOU Threshold override"
pipeline.spec.description.input[2].shortDescription = "(optional) Confidence Threshold override"
pipeline.spec.description.output[0].shortDescription = u"Boxes \xd7 Class confidence"
pipeline.spec.description.output[1].shortDescription = u"Boxes \xd7 [x, y, width, height] (relative to image size)"

pipeline.spec.description.metadata.versionString = "ssdlite_mobilenet_v2_coco_2018_05_09"
pipeline.spec.description.metadata.shortDescription = "MobileNetV2 + SSDLite, trained on COCO"
pipeline.spec.description.metadata.author = "Converted to Core ML by Matthijs Hollemans"
pipeline.spec.description.metadata.license = "https://github.com/tensorflow/models/blob/master/research/object_detection"
user_defined_metadata = {
    "classes": ",".join(labels),
    "iou_threshold": str(default_iou_threshold),
    "confidence_threshold": str(default_confidence_threshold)
}
pipeline.spec.description.metadata.userDefined.update(user_defined_metadata)
pipeline.spec.specificationVersion = 3
final_model = coremltools.models.MLModel(pipeline.spec)
final_model.save(coreml_model_path)
