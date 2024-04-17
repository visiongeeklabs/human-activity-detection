import tensorflow.compat.v1 as tf
from google.colab.patches import cv2_imshow
import numpy as np
import time
import cv2

tf.disable_v2_behavior()

tf.test.is_gpu_available()

frozen_graph_path = "frozen_inference_graph.pb"
img_path = "person_standing.jpg"

image = cv2.imread(img_path)
h, w, c = image.shape
print(h, w, c)
image_np = np.expand_dims(image, axis=0)

with open("labels.txt", 'r') as f:
  labels = [line.strip() for line in f.readlines()]

print(len(labels))

with tf.Graph().as_default():
  graph_def = tf.GraphDef()

  t1 = time.time()
  with tf.gfile.GFile(frozen_graph_path, 'rb') as frozen_graph:
    serialized_graph = frozen_graph.read()
    graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(graph_def, name='')
  t2 = time.time()
  print("model loading time: ", t2 - t1)

  with tf.Session() as sess:
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}

    for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes']:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                tensor_name)

    image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
    #print(tensor_dict)

    start = time.time()
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_np})
    finish = time.time()

    print("inference time: ", finish - start)
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    #print(output_dict)
    out = image.copy()
    threshold = 0.5
    for i in range(output_dict['num_detections']):
      if int(output_dict['detection_classes'][i]) not in [1,3,17,37,43,45,46,47,59,65,74,77,78,79,80]:
        if output_dict['detection_scores'][i] > threshold:
          print(labels[int(output_dict['detection_classes'][i])-1])
          print(output_dict['detection_scores'][i])
          bbox = output_dict['detection_boxes'][i]
          print(bbox)
          bbox[0] *= h
          bbox[1] *= w
          bbox[2] *= h
          bbox[3] *= w
          print((int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])))
          cv2.rectangle(out, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (0,255,0), 2)
          cv2.putText(out, labels[int(output_dict['detection_classes'][i])-1], (int(bbox[1]),int(bbox[0]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    cv2_imshow(out)
