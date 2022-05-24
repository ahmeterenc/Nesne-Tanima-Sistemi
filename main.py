import streamlit as st

import moviepy.editor as moviepy


from image import object_detection_image
from video import object_detection_video
from webcamera import object_detection_webcamera

def main():
    st.header('Nesne Tanıma Sistemine hoş geldiniz.')
    st.title("OpenCV ve Streamlit kullanılarak gerçekleştirilmiştir.")

    st.sidebar.title('Geçiş Yapmak istediğiniz bölüm seçiniz.')
    choice = st.sidebar.selectbox('Bölüm', ('Hakkında', 'Resim', 'Video','Web Kamera'))
    if choice == 'Resim':
        object_detection_image()

    elif choice == 'Video':
        object_detection_video()

        try:
            clip = moviepy.VideoFileClip('detected_video.mp4')
            clip.write_videofile('myvideo.mp4')
            st_video = open('myvideo.mp4', 'rb')
            video_bytes = st_video.read()
            st.write('Saptanmış Video')
        except OSError:
            ''
    elif choice == 'Web Kamera':
        object_detection_webcamera()

    elif choice == 'Hakkında':
        print()


if __name__ == '__main__':
    main()


