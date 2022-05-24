import streamlit as st
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import os
# import tensorflow as tf
# import tensorflow_hub as hub
import time
import urllib.request
import urllib
import moviepy.editor as moviepy
import cv2
import time
import sys


def object_detection_video():
    # object_detection_video.has_beenCalled = True
    # pass
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5
    config_path = r'yolov4.cfg'
    weights_path = r'yolov4.weights'
    font_scale = 1
    thickness = 1
    url = "https://raw.githubusercontent.com/zhoroh/ObjectDetection/master/labels/coconames.txt"
    f = urllib.request.urlopen(url)
    labels = [line.decode('utf-8').strip() for line in f]

    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

    ln = net.getLayerNames()
    ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    st.title("Video için Nesne Tanıma Sistemi")
    st.subheader("""
    Bu nesne tanıma sistemi bir videoyu alır ve videodaki nesnelerin etrafında oluşturulan sınırlayıcı kutularla videoyu çıkarır. 
    """
                 )
    uploaded_video = st.file_uploader("Yüklenen video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

        vid = uploaded_video.name
        with open(vid, mode='wb') as f:
            f.write(uploaded_video.read())  # videoyu diske kaydet

        st_video = open(vid, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Yüklenen video")

        cap = cv2.VideoCapture(vid)
        _, image = cap.read()
        h, w = image.shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mpv4')
        out = cv2.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))
        count = 0
        while True:
            _, image = cap.read()
            if _ != False:
                h, w = image.shape[:2]
                blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
                net.setInput(blob)
                start = time.perf_counter()
                layer_outputs = net.forward(ln)
                time_took = time.perf_counter() - start
                count += 1
                print(f"Time took: {count}", time_took)
                boxes, confidences, class_ids = [], [], []

                # seviye çıktılarının her biri üzerinde döngü
                for output in layer_outputs:
                    # nesne tanımalarının her biri üzerinde döngü
                    for detection in output:
                        # sınıf kimliğini (etiketini) ve güvenini (olasılık olarak) çıkarır
                        # the current object detection
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        # tespit edilmesini sağlayarak zayıf tahminleri atar
                        # olasılık, minimum olasılıktan daha büyük
                        if confidence > CONFIDENCE:
                            # sınırlayıcı kutu koordinatlarını göreli olarak geri ölçeklendir
                            # görüntünün boyutu, YOLO'nun aslında bulunur
                            # sınırlamanın merkez (x, y) koordinatlarını döndürür
                            # kutunun ardından kutuların genişliği ve yüksekliği verilir
                            box = detection[:4] * np.array([w, h, w, h])
                            (centerX, centerY, width, height) = box.astype("int")

                            # tepeyi elde etmek için merkez (x, y) koordinatlarını ve sınırlayıcı kutunun sol köşesini kullanın
                            x = int(centerX - (width / 2))
                            y = int(centerY - (height / 2))

                            # sınırlayıcı kutu koordinatlarını, güvenilirlik listesini ve sınıf kimliklerini güncelleyin
                            boxes.append([x, y, int(width), int(height)])
                            confidences.append(float(confidence))
                            class_ids.append(class_id)


                idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

                font_scale = 0.6
                thickness = 1
                # daha önce tanımlanan puanlar göz önüne alındığında maksimum olmayan bastırmayı gerçekleştirin
                if len(idxs) > 0:
                    # tuttuğumuz dizinler üzerinde döngü
                    for i in idxs.flatten():
                        # sınırlayıcı kutu koordinatlarını çıkar
                        x, y = boxes[i][0], boxes[i][1]
                        w, h = boxes[i][2], boxes[i][3]
                        # resmin üzerine bir sınırlayıcı kutu dikdörtgeni çizin ve etiketleyin
                        color = [int(c) for c in colors[class_ids[i]]]
                        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
                        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
                        # şeffaf kutuları metnin arka planı olarak çizmek için metin genişliğini ve yüksekliğini hesaplayın
                        (text_width, text_height) = \
                            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[
                                0]
                        text_offset_x = x
                        text_offset_y = y - 5
                        box_coords = (
                            (text_offset_x, text_offset_y),
                            (text_offset_x + text_width + 2, text_offset_y - text_height))
                        overlay = image.copy()
                        cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
                        #opaklık ekle (kutuya şeffaflık)
                        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
                        # şimdi metni koyun (etiket: güven %)
                        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                    fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

                out.write(image)
                cv2.imshow("Resim", image)

                if ord("q") == cv2.waitKey(1):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()
