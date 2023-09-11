from word_detector import detect, prepare_img, sort_multiline
from path import Path
import matplotlib.pyplot as plt
import cv2
import argparse
import os


class ImageProcessor:
    def __init__(self, data_dir, kernel_size=25, sigma=11, theta=7, min_area=100, img_height=1000):
        self.data_dir = data_dir
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta
        self.min_area = min_area
        self.img_height = img_height
        self.list_img_names_serial = []

    def get_img_files(self):
        res = []
        for ext in ['*.png', '*.jpg', '*.bmp']:
            res += Path(self.data_dir).files(ext)
        return res

    def process_images(self):
        for fn_img in self.get_img_files():
            img = prepare_img(cv2.imread(fn_img), self.img_height)
            detections = detect(img,
                                kernel_size=self.kernel_size,
                                sigma=self.sigma,
                                theta=self.theta,
                                min_area=self.min_area)

            lines = sort_multiline(detections)

            plt.imshow(img, cmap='gray')
            num_colors = 7
            colors = plt.colormaps.get_cmap('rainbow')
            for line_idx, line in enumerate(lines):
                for word_idx, det in enumerate(line):
                    xs = [det.bbox.x, det.bbox.x, det.bbox.x +
                          det.bbox.w, det.bbox.x + det.bbox.w, det.bbox.x]
                    ys = [det.bbox.y, det.bbox.y + det.bbox.h,
                          det.bbox.y + det.bbox.h, det.bbox.y, det.bbox.y]
                    plt.plot(xs, ys, c=colors(line_idx % num_colors))
                    plt.text(det.bbox.x, det.bbox.y, f'{line_idx}/{word_idx}')
                    crop_img = img[det.bbox.y:det.bbox.y + det.bbox.h, det.bbox.x:det.bbox.x + det.bbox.w]

                    path = './test_images'
                    isExist = os.path.exists(path)
                    if not isExist:
                        os.mkdir(path)
                        print("Directory Created")

                    cv2.imwrite(f"{path}/line" + str(line_idx) + "word" + str(word_idx) + ".jpg", crop_img)
                    full_img_path = "line" + str(line_idx) + "word" + str(word_idx) + ".jpg"
                    self.list_img_names_serial.append(full_img_path)
                    list_img_names_serial_set = set(self.list_img_names_serial)

                    textfile = open("./examples/img_names_sequence.txt", "w")
                    for element in self.list_img_names_serial:
                        textfile.write(element + "\n")
                    textfile.close()

            plt.show()
            return path


def detect_path(image_path):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(image_path))
    parser.add_argument('--kernel_size', type=int, default=25)
    parser.add_argument('--sigma', type=float, default=11)
    parser.add_argument('--theta', type=float, default=7)
    parser.add_argument('--min_area', type=int, default=100)
    parser.add_argument('--img_height', type=int, default=1000)
    parsed = parser.parse_args()

    processor = ImageProcessor(parsed.data, parsed.kernel_size, parsed.sigma, parsed.theta, parsed.min_area, parsed.img_height)
    path = processor.process_images()
    return path


if __name__ == "__main__":
    pass
