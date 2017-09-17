# -*- coding:utf-8 -*-


import cv2
import os
import argparse
import sys


def video2image():
    cap = cv2.VideoCapture(args.videopath)

    i = 0
    name = 0
    while cap.isOpened():
        ret, frame = cap.read()
        time = cap.get(cv2.CAP_PROP_POS_MSEC)
        index = cap.get(cv2.CAP_PROP_POS_FRAMES)
        print('frames: %d   ---   times: %f' % (index, time/1000))
        if frame is None:
            break
        i += 1
        if args.videofps <= 0:
            cv2.imwrite(os.path.join(args.imagepath, str(name)) + '.jpg', frame)
            name += 1
            print('(height: %d, weight: %d, channel: %d)' % frame.shape)
        else:
            if i == args.videofps:
                # cv2.imshow('frame', frame)
                # k = cv2.waitKey(20)
                # k = cv2.waitKey(0)
                i = 0
                cv2.imwrite(os.path.join(args.imagepath, str(name)) + '.jpg', frame)
                name += 1
                print('(height: %d, weight: %d, channel: %d)' % frame.shape)

    cap.release()
    cv2.destroyAllWindows()


def image2video():
    video_writer = cv2.VideoWriter(filename=args.videopath,
                                   fourcc=cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps=args.videofps, frameSize=(videosize[0], videosize[1]))
    list = os.listdir(args.imagepath)
    list.sort()
    for jpg in list:
        video_writer.write(cv2.imread(os.path.join(args.imagepath, jpg)))
    video_writer.release()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='video args',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--videopath', type=str, default='/home/workspace/DATASET/labixiaoxin2.flv',
                        help='path of video')
    parser.add_argument('--videosize', type=str, default='800, 600',
                        help='frame size of generated video')
    parser.add_argument('--videofps', type=int, default=24,
                        help='fps of generated video or rate image from video')
    parser.add_argument('--imagepath', type=str, default='image',
                        help='path of image')
    parser.add_argument('--ope', type=str, default='2img', help='2img for video to image,2video for image to video')

    args = parser.parse_args()

    videosize = [int(l) for l in args.videosize.split(',')]

    if args.ope == '2img':
        if not os.path.exists(args.imagepath):
            os.mkdir(args.imagepath)
        video2image()
    elif args.ope == '2video':
        if not os.path.exists(args.videopath):
            os.mkdir(args.videopath)
        image2video()
