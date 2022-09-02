import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import math
import cv2
import os
import shutil
import tensorflow as tf
from utils import label_map_util


option = 1

from digi.xbee.devices import XBeeDevice
from digi.xbee.devices import RemoteXBeeDevice
from digi.xbee.devices import XBee64BitAddress


# Ensure that data collection directory exists, reset data collection for new run

if not os.path.exists("data"):
    os.mkdir("data")

if not os.path.exists("data/option{}".format(option)):
    os.mkdir("data/option{}".format(option))

if os.path.exists("data/option{}/frames".format(option)):
    shutil.rmtree('data/option{}/frames'.format(option))

if os.path.exists("data/option{}/plots".format(option)):
    shutil.rmtree('data/option{}/plots'.format(option))

os.mkdir('data/option{}/frames'.format(option))
os.mkdir('data/option{}/plots'.format(option))

# Defining paths to label map and frozen inference graph for object detection

PATH_TO_FROZEN_GRAPH = 'frozen_inference_graph.pb'
PATH_TO_LABELS = 'label_map.pbtxt'


iterator = 0


device = XBeeDevice("COM7", 9600) # Pick your specific port

# Open Controller Device

device.open()

remote_devices = []

remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A20040D835D1")))
#remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200418FE7A3")))
#remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200415E8441")))
#remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A2004190B158")))
#remote_devices.append(RemoteXBeeDevice(device, XBee64BitAddress.from_hex_string("0013A200417CA46A")))

# Define frozen inference graph from path

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.compat.v1.GraphDef()
  with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

# Define label map from path

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

#discretized space
vidcap = cv2.VideoCapture(0)
success, frame = vidcap.read()

width_sections = 64
height_sections = 48

xdim = np.linspace(5, frame.shape[1] - 5, width_sections)
ydim = np.linspace(5, frame.shape[0] - 5, height_sections)
X, Y = np.meshgrid(xdim,ydim)




Δ = 0.01

x_g=0.8
y_g=0.6

x_k=0
y_k=0

x_error=x_k-x_g
y_error=y_k-y_g

tf=20
NN=tf / Δ
α = 0.2
γ = 0.9
R = [[0.05, 0], [0, 0.05]]
q = [[1, 0], [0, 1]]



def phi_basis(X1, X2):
    result = np.array([[X1, X2, X1*X1, X1*X2, X2*X2]]).T

    return result

def D_phi_basis(X1, X2):
    result = np.array([[1, 0, 2*X1, X2, 0],
                       [0, 1, 0, X1, 2*X2]]).T
    return (result)

NL = 5

W0 = np.array([[np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1)]]).T

U0 = np.array([[np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1),
                np.random.normal(0, 0.1)]]).T

def f(x_error, y_error):
    result = np.array([[x_error],[y_error]])

    return result

def g(u):
    result = np.array([Δ,Δ])
    return result

def r(X1, X2, u):
    X = np.array([[X1],
                  [X2]])
    result = X.T @ q @ X + u.T @ R @ u
    return (result)


X_list = []
U_list = []
W_list = [W0]
u_list = []
u_k=1
W_k = W0
U_k = U0
kt = 0
uj_x = (-γ / 2)* np.linalg.inv(R) @ g(u_k) @ D_phi_basis(x_error, y_error).T @ W0
uj_y=uj_x
u_j=np.array([uj_x, uj_y])

Wji = W0

tc = []
x_list, y_list = [],[]

w1_list, w2_list, w3_list, w4_list, w5_list, thetades_list = [],[],[],[],[],[]

rlist=[]

prevdes=0

#pre Initialization and initialization



for k in range(int(NN)):

    # Start object detection

    with detection_graph.as_default() as graph:
        with tf.compat.v1.Session() as sess:

                        for i in range(0, len(remote_devices)):

                              success, frame = vidcap.read()

                              frame_expanded = np.expand_dims(frame, axis=0)

                              output_dict = function.run_inference_for_single_image(frame_expanded, graph, sess) # Detect Alphabot2 in the frame

                              rects = []

                              # Draw rectangles around all Alphabot2 robots detected

                              for j in range(0, len(remote_devices)):
                                if output_dict['detection_scores'][j] > 0.85:

                                      (startY, startX, endY, endX) = output_dict['detection_boxes'][j]

                                      rects.append([int(startX*frame.shape[1]), int(startY*frame.shape[0]), int(endX*frame.shape[1]), int(endY*frame.shape[0])])

                                      cv2.rectangle(frame, (int(startX*frame.shape[1]), int(startY*frame.shape[0])), (int(endX*frame.shape[1]), int(endY*frame.shape[0])),
                                         (0, 255, 0), 1)

                              objects = tracker.update(rects) # Begin tracking all Alphabot2 robots that are detected

                              for (objectID, centroid) in objects.items():
                                  text = "ID {}".format(objectID)
                                  cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                  cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                                  positionInit[objectID] = centroid

                              cv2.imshow("Frame", frame)
                              cv2.waitKey(1)
    x_k=centroid[0]
    y_k=centroid[1]

    x_list.append(x_error)
    y_list.append(y_error)
    X_k = f(x_error, y_error) + g(u_k)*u_j
    X_list.append(X_k)
    x_error, y_error  = X_k[0][0], X_k[1][0]

    phi = phi_basis(x_error, y_error)

    r_k = r(x_error, y_error, u_j) + γ*float(phi_basis(x_error, y_error).T @ Wji)

    rlist.append(r_k)

    ecount = 1
    Wjim = W_list[-1]
    Wji = Wji - α*phi_basis(x_error, y_error)*(float(phi_basis(x_error, y_error).T @ Wji) - r_k)

    while ecount < 100 and float((Wji - Wjim).T @ (Wji - Wjim)) > 10**-5:
        Wjim = W_list[-1]
        #gradient descent
        Wji = Wji - α*phi_basis(x_error, y_error)*(float(phi_basis(x_error, y_error).T @ Wji) - r_k)
        ecount+=1


    W_list.append(Wji)

    w1_list.append(Wji[0][0])
    w2_list.append(Wji[1][0])
    w3_list.append(Wji[2][0])
    w4_list.append(Wji[3][0])
    w5_list.append(Wji[4][0])


    # find desired angle
    #prevdes should be a measurement of the new orientation


    thetades=np.arctan2(u_j[1][0], u_j[0][0])
    currentdes=thetades
    thetades_list.append(thetades)

    #use optimal control to move to desried angle

    message = "P,((currentdes-prevdes)*180/(np.pi))>"
    device.send_data_async(remote_devices[iterator], message)
    print(message)

    prevdes=currentdes


    #move forward with found magnitude sqrt(u_j[1][0]*u_j[1][0]+u_j[0][0]*u_j[0][0])

    v_mag=sqrt(u_j[1][0]**2+u_j[0][0]**2)
    v_arduino=v_mag*255
    message = "Z,v_arduino>"
    device.send_data_async(remote_devices[iterator], message)
    print(message)


    #need to update u with the weights from W
    #policy improvement - update u_j

    u_j = -(γ/2)*np.linalg.inv(R) * g(u_k) @ D_phi_basis(x_error, y_error).T @ W_list[-1]

    u_list.append(u_j)
    tc.append(k)



# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], x_list[start_idx:end_idx], label = "x_error")
# plt.plot(tc[start_idx:end_idx], y_list[start_idx:end_idx], label = "y_error")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()


# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], w1_list[start_idx:end_idx], label = "w1")
# plt.plot(tc[start_idx:end_idx], w2_list[start_idx:end_idx], label = "w2")
# plt.plot(tc[start_idx:end_idx], w3_list[start_idx:end_idx], label = "w3")
# plt.plot(tc[start_idx:end_idx], w4_list[start_idx:end_idx], label = "w4")
# plt.plot(tc[start_idx:end_idx], w5_list[start_idx:end_idx], label = "w5")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()


# def error_to_trajectory(error_list):
#     result = []
#     for er in error_list:
#         result.append(er+x_g)
#     return result
# x_list = error_to_trajectory(x_list)
# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], x_list[start_idx:end_idx], label = "x")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()


# def error_to_trajectory(error_list):
#    result = []
#    for er in error_list:
#        result.append(er+y_g)
#    return result
# y_list = error_to_trajectory(y_list)
# start_idx = 0
# end_idx = int(NN)
# plt.plot(tc[start_idx:end_idx], y_list[start_idx:end_idx], label = "y")
# plt.xlabel("timestep")
# plt.ylabel("parameter")
# plt.legend()
