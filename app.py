
# Author: vlarobbyk
# Version: 1.0
# Date: 2024-10-20
# Description: A simple example to process video captured by the ESP32-XIAO-S3 or ESP32-CAM-MB in Flask.


from flask import Flask, render_template, Response, stream_with_context, Request
from io import BytesIO

import cv2
import numpy as np
import requests
import os
import time
import random

app = Flask(__name__)
# IP Address
_URL = 'http://192.168.18.136'
# Default Streaming Port
_PORT = '81'
# Default streaming route
_ST = '/stream'
SEP = ':'

stream_url = ''.join([_URL,SEP,_PORT,_ST])

images_folder = 'static/imgs'

processed_images = []

salt_level = 100
pepper_level = 100


    
for filename in os.listdir(images_folder):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.webp')):
        image_path = os.path.join(images_folder, filename)
        processed_images.append(image_path)

bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=25, varThreshold=20, detectShadows=True)
use_bg_subtraction = False

modes = ["original", "bg_subtraction", "histogram_eq", "clahe", "homomorphic", "salt_and_peper_noise", "border-without-filter", "border-with-median", "border-with-gaussian", "border-with-blur"]
current_mode_index = 0

def video_capture():
    frame_count = 0
    start_time = time.time()
    while True:
        try:
            res = requests.get(stream_url, stream=True, timeout=5)
            for chunk in res.iter_content(chunk_size=100000):
                if len(chunk) > 100:
                    img_data = BytesIO(chunk)
                    cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
                    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

                    display_img = gray

                    if current_mode_index == 0:
                        display_img = cv_img.copy()
                    elif current_mode_index == 1:
                        fg_mask = bg_subtractor.apply(gray)
                        display_img = fg_mask
                    elif current_mode_index == 2:
                        display_img = cv2.equalizeHist(gray)
                    elif current_mode_index == 3:
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        display_img = clahe.apply(gray)
                    elif current_mode_index == 4:
                        display_img = homomorphic_filter(gray)
                    elif current_mode_index == 5:
                        noisy_img = salt_pepper_noise(cv_img.copy(), salt_level, pepper_level)
                        median_filtered = median_filter(noisy_img)
                        gaussian_filtered = gaussian_filter(noisy_img)
                        blurred = apply_blur(noisy_img)

                        top_row = np.hstack((noisy_img, median_filtered))
                        bottom_row = np.hstack((gaussian_filtered, blurred))
                        display_img = np.vstack((top_row, bottom_row))

                    elif current_mode_index == 6:
                        canny_edge = canny_edge_detection(gray)
                        sobel_edge = sobel_edge_detection(gray)
                        display_img = np.hstack((canny_edge, sobel_edge))
                    elif current_mode_index == 7:
                        median_filtered = median_filter(gray)
                        canny_edge = canny_edge_detection(median_filtered)
                        sobel_edge = sobel_edge_detection(median_filtered)
                        display_img = np.hstack((canny_edge, sobel_edge))
                    elif current_mode_index == 8:
                        gaussian_filtered = gaussian_filter(gray)
                        canny_edge = canny_edge_detection(gaussian_filtered)
                        sobel_edge = sobel_edge_detection(gaussian_filtered)
                        display_img = np.hstack((canny_edge, sobel_edge))
                    elif current_mode_index == 9:
                        blurred = apply_blur(gray)
                        canny_edge = canny_edge_detection(blurred)
                        sobel_edge = sobel_edge_detection(blurred)
                        display_img = np.hstack((canny_edge, sobel_edge))
                        

                    # Añadir FPS al frame
                    frame_count += 1
                    elapsed_time = time.time() - start_time
                    fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                    cv2.putText(display_img, f'FPS: {fps:.2f}', (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                    # Codificar imagen para transmisión
                    flag, encodedImage = cv2.imencode(".jpg", display_img)
                    if not flag:
                        continue

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' +
                           bytearray(encodedImage) + b'\r\n')

                    if elapsed_time >= 1.0:
                        frame_count = 0
                        start_time = time.time()

        except requests.exceptions.RequestException as e:
            print("Error en la conexión:", e)
            continue 


def canny_edge_detection(img):
    return cv2.Canny(img, 100, 100*1.3, 3)

def sobel_edge_detection(img):
    sobel_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    return np.uint8(sobel_magnitude)

def salt_pepper_noise(img, salt_level, pepper_level):
    if len(img.shape) == 3:
        height, width, _ = img.shape
    else:
        height, width = img.shape

    xs = np.random.randint(0, width, salt_level)
    ys = np.random.randint(0, height, salt_level)
    xp = np.random.randint(0, width, pepper_level)
    yp = np.random.randint(0, height, pepper_level)

    if len(img.shape) == 3:
        img[ys, xs, :] = 255  # Sal
        img[yp, xp, :] = 0    # Pimienta
    else:
        img[ys, xs] = 255
        img[yp, xp] = 0

    return img

def median_filter(img):
    if len(img.shape) == 3:
        channels = cv2.split(img)
        channels = [cv2.medianBlur(channel, 15) for channel in channels]
        return cv2.merge(channels)
    else:
        return cv2.medianBlur(img, 15)

def gaussian_filter(img):
    ksize = (5, 5)
    sigma = 1.5
    if len(img.shape) == 3:
        channels = cv2.split(img)
        channels = [cv2.GaussianBlur(channel, ksize, sigma) for channel in channels]
        return cv2.merge(channels)
    else:
        return cv2.GaussianBlur(img, ksize, sigma)

def apply_blur(img):
    ksize = (9, 9)
    if len(img.shape) == 3:
        channels = cv2.split(img)
        channels = [cv2.blur(channel, ksize) for channel in channels]
        return cv2.merge(channels)
    else:
        return cv2.blur(img, ksize)


def homomorphic_filter(image, low_gamma=0.5, high_gamma=2.0, sigma=30):
    scale = 0.5
    small_image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
    image_log = np.log1p(np.array(small_image, dtype="float"))

    dft = np.fft.fft2(image_log)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = small_image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), dtype=np.float32)
    for i in range(rows):
        for j in range(cols):
            d = np.sqrt((i - crow) ** 2 + (j - ccol) ** 2)
            mask[i, j] = (high_gamma - low_gamma) * (1 - np.exp(-((d ** 2) / (2 * (sigma ** 2))))) + low_gamma

    dft_shift_filtered = dft_shift * mask
    dft_inverse = np.fft.ifftshift(dft_shift_filtered)
    img_back = np.fft.ifft2(dft_inverse)
    img_back = np.expm1(np.abs(img_back))

    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
    img_back = np.array(img_back, dtype="uint8")
    img_back = cv2.resize(img_back, (image.shape[1], image.shape[0]))

    return img_back

def calculate_cir(original_image, enhanced_image):
    inner_window_size = 3
    outer_window_size = 7

    inner_window_orig = cv2.boxFilter(original_image.astype(np.float32), -1, (inner_window_size, inner_window_size))
    outer_window_orig = cv2.boxFilter(original_image.astype(np.float32), -1, (outer_window_size, outer_window_size))
    
    inner_window_enh = cv2.boxFilter(enhanced_image.astype(np.float32), -1, (inner_window_size, inner_window_size))
    outer_window_enh = cv2.boxFilter(enhanced_image.astype(np.float32), -1, (outer_window_size, outer_window_size))

    contrast_orig = np.abs(inner_window_orig - outer_window_orig) / (inner_window_orig + outer_window_orig + 1e-5)
    contrast_enh = np.abs(inner_window_enh - outer_window_enh) / (inner_window_enh + outer_window_enh + 1e-5)

    cir_value = np.sum((contrast_enh - contrast_orig) ** 2) / (np.sum(contrast_orig ** 2) + 1e-5)
    return cir_value

def apply_morphology(images, initial_radius=30, num_iterations=5):
    for image_path in images:
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        output_dir = os.path.join('static', image_name)
        os.makedirs(output_dir, exist_ok=True)

        erode_dir = os.path.join(output_dir, 'erosion')
        dilate_dir = os.path.join(output_dir, 'dilation')
        top_hat_dir = os.path.join(output_dir, 'top_hat')
        black_hat_dir = os.path.join(output_dir, 'black_hat')
        contrast_enhanced_dir = os.path.join(output_dir, 'contrast_enhanced')
        best_result_dir = os.path.join(output_dir, 'best_contrast_enhanced')
        os.makedirs(erode_dir, exist_ok=True)
        os.makedirs(dilate_dir, exist_ok=True)
        os.makedirs(top_hat_dir, exist_ok=True)
        os.makedirs(black_hat_dir, exist_ok=True)
        os.makedirs(contrast_enhanced_dir, exist_ok=True)
        os.makedirs(best_result_dir, exist_ok=True)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        best_cir = -1
        best_result = None
        best_suffix = ""

        for i in range(1, num_iterations + 1):
            kernel_size = initial_radius + i * 4 
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            
            # Aplicar transformaciones
            erode = cv2.erode(image, kernel)
            dilate = cv2.dilate(image, kernel)
            top_hat_image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
            black_hat_image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
            contrast_enhanced_image = cv2.add(image, cv2.subtract(top_hat_image, black_hat_image))

            cir_value = calculate_cir(image, contrast_enhanced_image)
            print(f"CIR para {image_name} en iteración {i} (radio {kernel_size}) y CLI {cir_value}")

            suffix = f"iter_{i}_kernel_{kernel_size}x{kernel_size}"
            cv2.imwrite(os.path.join(erode_dir, f'erosion_{suffix}.png'), erode)
            cv2.imwrite(os.path.join(dilate_dir, f'dilation_{suffix}.png'), dilate)
            cv2.imwrite(os.path.join(top_hat_dir, f'top_hat_{suffix}.png'), top_hat_image)
            cv2.imwrite(os.path.join(black_hat_dir, f'black_hat_{suffix}.png'), black_hat_image)
            cv2.imwrite(os.path.join(contrast_enhanced_dir, f'contrast_enhanced_{suffix}.png'), contrast_enhanced_image)

            if cir_value > best_cir:
                best_cir = cir_value
                best_result = contrast_enhanced_image.copy()
                best_suffix = suffix

        if best_result is not None:
            cv2.imwrite(os.path.join(best_result_dir, f'best_contrast_enhanced_{best_suffix}.png'), best_result)
            print(f"Mejor resultado para {image_name}: iteración {best_suffix} con CIR {best_cir}")

apply_morphology(processed_images)

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_video_mode/<int:mode_index>")
def set_video_mode(mode_index):
    global current_mode_index
    if 0 <= mode_index < len(modes):
        current_mode_index = mode_index
        print(f"Modo de video cambiado a: {modes[current_mode_index]}")
        return '', 204 
    else:
        return "Índice de modo no válido", 400  

@app.route('/set_salt_level/<int:level>', methods=['GET'])
def set_salt_level(level):
    global salt_level
    salt_level = level
    return "Nivel de sal y pimienta ajustado a " + str(level)

@app.route('/set_pepper_level/<int:level>', methods=['GET'])
def set_pepper_level(level):
    global pepper_level
    pepper_level = level
    return "Nivel de sal y pimienta ajustado a " + str(level)

if __name__ == "__main__":
    app.run(debug=True)

