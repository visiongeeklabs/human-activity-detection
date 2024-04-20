import numpy as np
import time
import sys
import cv2
import tensorflow.compat.v1 as tf

PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'labels.txt'

with open(PATH_TO_LABELS, 'r') as f:
  labels = [line.strip() for line in f.readlines()]

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.GraphDef()

    fid = tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb')
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

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


video = cv2.VideoCapture(sys.argv[1])

if not video.isOpened():
    print("Could not open video")
    exit()

width = video.get(cv2.CAP_PROP_FRAME_WIDTH)
height = video.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = video.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
writer = cv2.VideoWriter('detection_output.mp4', fourcc, int(fps), (int(width),int(height)))


while video.isOpened():

    status, image = video.read()

    if not status:
        break
    
    image_exp = np.expand_dims(image, axis=0)
    start = time.time()
    output_dict = sess.run(tensor_dict, feed_dict={image_tensor: image_exp})
    finish = time.time()
    print("inference time: ", finish - start)

    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
      'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    out = image.copy()
    threshold = 0.5
    for i in range(output_dict['num_detections']):
        if int(output_dict['detection_classes'][i]) not in [1,3,17,37,43,45,46,47,59,65,74,77,78,79,80]:
            if output_dict['detection_scores'][i] > threshold:
                print(labels[int(output_dict['detection_classes'][i])-1])
                print(output_dict['detection_scores'][i])
                bbox = output_dict['detection_boxes'][i]
                print(bbox)
                bbox[0] *= height
                bbox[1] *= width
                bbox[2] *= height
                bbox[3] *= width
                print((int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])))
                cv2.rectangle(out, (int(bbox[1]), int(bbox[0])), (int(bbox[3]), int(bbox[2])), (0,255,0), 2)
                cv2.putText(out, labels[int(output_dict['detection_classes'][i])-1], (int(bbox[1]),int(bbox[0]-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    writer.write(out)
    
video.release()
cv2.destroyAllWindows()
writer.release()
sess.close()