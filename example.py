##
# coding:utf-8
##

import os
import glob
import argparse
import cv2
from pathlib import Path
from api import Detector

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str,
                        default='/media/wakaba/41572840-475d-437a-8ebf-be3ac4a42f86/R0010031')
    parser.add_argument('--exe', type=str, default='png')
    parser.add_argument('--weight', type=str,
                        default='./weights/pL1_MWHB1024_Mar11_4000.ckpt')
    parser.add_argument('--output', type=str,
                        default='./output_R0010031')
    return parser.parse_args()


if __name__=="__main__":
    args = argparser()

    # Initialize detector
    detector = Detector(model_name='rapid',
                        weights_path=args.weight)

    if not os.path.isdir(args.output):
        os.makedirs(args.output)


    img_list = glob.glob(os.path.join(args.img_path, f"*.{args.exe}"))
    img_list.sort()       

    for img_path in img_list:
        print(img_path)
        # A simple example to run on a single image and plt.imshow() it
        detect_result = detector.detect_one(img_path=img_path,
                        input_size=1024, conf_thres=0.3,
                        visualize=False)

        json_name = Path(img_path).name.replace(f"{args.exe}", 'json')
        output_name = os.path.join(args.output, json_name)

        with open(output_name, 'w') as fp:
            fp.write(f"{detect_result.tolist()}")
