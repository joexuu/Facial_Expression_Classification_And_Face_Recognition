# 随机生成训练集和测试集
from __future__ import print_function
import random, shutil
import numpy as np
import os
import torch.backends.cudnn as cudnn
from PIL import Image
import torch
import argparse
from data import cfg_mnet, cfg_re50
from models.retinaface import RetinaFace
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.box_utils import decode, decode_landm
import time
import cv2
import math
from collections import defaultdict

def movefile(ori_path,split_Dir):
    for name in os.listdir(ori_path):
        labels=int(emotions[name.split('_')[4]])
        if labels == 0:
           split_Dir2 = os.path.join(split_Dir, '0/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        if labels == 1:
           split_Dir2 = os.path.join(split_Dir, '1/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        if labels == 2:
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
           split_Dir2 = os.path.join(split_Dir, '2/')
        if labels == 3:
           split_Dir2 = os.path.join(split_Dir, '3/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        if labels == 4:
           split_Dir2 = os.path.join(split_Dir, '4/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        if labels == 5:
           split_Dir2 = os.path.join(split_Dir, '5/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        if labels == 6:
           split_Dir2 = os.path.join(split_Dir, '6/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        if labels == 7:
           split_Dir2 = os.path.join(split_Dir, '7/')
           shutil.move(os.path.join(ori_path, name), os.path.join(split_Dir2, name))
        print(name)
    return

def movesplitFile(ori, test):
    pathDir=os.listdir(ori)  # 取图片的原始路径
    filenumber=len(pathDir)
    picknumber=int(filenumber * ratio)  # 按照rate比例从文件夹中取一定数量图片
    print(picknumber,filenumber)
    sample=random.sample(pathDir, picknumber)  # 随机选取picknumber数量的样本图片\
    for name in sample:
        train = os.path.join(ori, name)
        shutil.move(train, test)
        print('train', train)
    return

def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

def align_face(image_array, landmarks):
    """ align faces according to eyes position
    :param image_array: numpy array of a single image
    :param landmarks: dic     t of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    rotated_img:  numpy array of aligned image
    eye_center: tuple of coordinates for eye center
    angle: degrees of rotation
    """
    # get list landmarks of left and right eye
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    # calculate the mean point of landmarks of left and right eye
    left_eye_center = np.mean(left_eye, axis=0).astype("int")
    right_eye_center = np.mean(right_eye, axis=0).astype("int")
    # compute the angle between the eye centroids
    dy = right_eye_center[1] - left_eye_center[1]
    dx = right_eye_center[0] - left_eye_center[0]
    # compute angle between the line of 2 centeroids and the horizontal line
    angle = math.atan2(dy, dx) * 180. / math.pi
    # calculate the center of 2 eyes
    eye_center = (int((left_eye_center[0] + right_eye_center[0] // 2)),
                  (int(left_eye_center[1] + right_eye_center[1] // 2)))
    # at the eye_center, rotate the image by the angle
    rotate_matrix = cv2.getRotationMatrix2D(eye_center, angle, scale=1)
    rotated_img = cv2.warpAffine(image_array, rotate_matrix, (image_array.shape[1], image_array.shape[0]))
    return rotated_img, eye_center, angle

def rotate(origin, point, angle, row):
    """ rotate coordinates in image coordinate system
    :param origin: tuple of coordinates,the rotation center
    :param point: tuple of coordinates, points to rotate
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated coordinates of point
    """
    x1, y1 = point
    x2, y2 = origin
    y1 = row - y1
    y2 = row - y2
    angle = math.radians(angle)
    x = x2 + math.cos(angle) * (x1 - x2) - math.sin(angle) * (y1 - y2)
    y = y2 + math.sin(angle) * (x1 - x2) + math.cos(angle) * (y1 - y2)
    y = row - y
    return int(x), int(y)

def rotate_landmarks(landmarks, eye_center, angle, row):
    """ rotate landmarks to fit the aligned face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param eye_center: tuple of coordinates for eye center
    :param angle: degrees of rotation
    :param row: row size of the image
    :return: rotated_landmarks with the same structure with landmarks, but different values
    """
    rotated_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            rotated_landmark = rotate(origin=eye_center, point=landmark, angle=angle, row=row)
            rotated_landmarks[facial_feature].append(rotated_landmark)
    return rotated_landmarks

def corp_face(image_array, landmarks):
    """ crop face according to eye,mouth and chin position
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return:
    cropped_img: numpy array of cropped image
    """

    eye_landmark = np.concatenate([np.array(landmarks['left_eye']),
                                   np.array(landmarks['right_eye'])])
    eye_center = np.mean(eye_landmark, axis=0).astype("int")
    lip_landmark = np.concatenate([np.array(landmarks['left_mouth']),
                                   np.array(landmarks['right_mouth'])])
    lip_center = np.mean(lip_landmark, axis=0).astype("int")
    mid_part = lip_center[1] - eye_center[1]
    top = eye_center[1] - mid_part * 30 / 35
    bottom = lip_center[1] + mid_part

    w = h = bottom - top
    x_min = np.min(landmarks['nose'], axis=0)[0]
    x_max = np.max(landmarks['nose'], axis=0)[0]
    x_center = (x_max - x_min) / 2 + x_min
    left, right = (x_center - w / 2, x_center + w / 2)

    pil_img = Image.fromarray(np.uint8(image_array))
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

def transform_image(root):
    for image_name in os.listdir(root):
        image_path=os.path.join(root, image_name)
        print(image_path)
        im0=cv2.imread(image_path)
        im0=cv2.copyMakeBorder(im0, 100, 100, 400, 400, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        resize=1

        img=np.float32(im0)
        print('im0.shape',img.shape)
        im_height, im_width, _=img.shape
        scale=torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img-=(104, 117, 123)
        img=img.transpose(2, 0, 1)
        img=torch.from_numpy(img).unsqueeze(0)
        img=img.to(device)
        scale=scale.to(device)

        tic=time.time()
        loc, conf, landms=net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))

        priorbox=PriorBox(cfg, image_size=(im_height, im_width))
        priors=priorbox.forward()
        priors=priors.to(device)
        prior_data=priors.data
        boxes=decode(loc.data.squeeze(0), prior_data, cfg['variance'])
        boxes=boxes * scale / resize
        boxes=boxes.cpu().numpy()
        scores=conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms=decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
        scale1=torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                             img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                             img.shape[3], img.shape[2]])
        scale1=scale1.to(device)
        landms=landms * scale1 / resize
        landms=landms.cpu().numpy()

        # ignore low scores
        inds=np.where(scores > args.confidence_threshold)[0]
        boxes=boxes[inds]
        landms=landms[inds]
        scores=scores[inds]

        # keep top-K before NMS
        order=scores.argsort()[::-1][:args.top_k]
        boxes=boxes[order]
        landms=landms[order]
        scores=scores[order]

        # do NMS
        dets=np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep=py_cpu_nms(dets, args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets=dets[keep, :]
        landms=landms[keep]

        # keep top-K faster NMS
        dets=dets[:args.keep_top_k, :]
        landms=landms[:args.keep_top_k, :]

        dets=np.concatenate((dets, landms), axis=1)

        # show image
        #if args.save_image:
        for b in dets:
            if b[4] < args.vis_thres:
                continue

            b=list(map(int, b))

            face_landmarks_dict={'left_eye': [(b[5], b[6])],
                                 'right_eye': [(b[7], b[8])],
                                 'nose': [(b[9], b[10])],
                                 'left_mouth': [(b[11], b[12])],
                                 'right_mouth': [(b[13], b[14])], }
            aligned_face, eye_center, angle=align_face(image_array=im0, landmarks=face_landmarks_dict)
            rotated_landmarks=rotate_landmarks(landmarks=face_landmarks_dict,
                                               eye_center=eye_center, angle=angle, row=im0.shape[0])
            cropped_face, left, top=corp_face(image_array=aligned_face, landmarks=rotated_landmarks)

            cropped_face=cropped_face[..., ::-1]
            img=Image.fromarray(cropped_face)
            img=img.resize((256, 256), Image.ANTIALIAS)
            os.remove(image_path)
            img.save(image_path)
    return

if __name__ == '__main__':
    ori_path='./data/RaFD1/' # 原始文件的路径
    split_Dir='./data/RaFD6/train/'  # 移动到新的文件夹路径
    test_Dir='./data/RaFD6/test/'
    emotions={'angry': 0, 'disgusted': 1, 'fearful': 2, 'happy': 3, 'sad': 4, 'surprised': 5, 'neutral': 6,
              'contemptuous': 7}

    parser=argparse.ArgumentParser(description='faceexpression')

    parser.add_argument('--top_k', default=5000, type=int, help='top_k')
    parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
    parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
    parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                        type=str, help='Trained state_dict file path to open')
    parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
    parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
    parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
    parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    args=parser.parse_args()

    ratio=0.1  # 抽取比例
    movefile(ori_path,split_Dir) #将原始图片按照7种表情分类移动到train文件夹
    for i in range(0,8): #将train文件夹下面的图片按照0.1的比例抽取到test对应的表情文件夹
        test_Dir2 = os.path.join(test_Dir, str(i))
        test_Dir2 = test_Dir2 + '/'
        split_Dir2=os.path.join(split_Dir, str(i))
        split_Dir2=split_Dir2 + '/'
        movesplitFile(split_Dir2, test_Dir2)

    torch.set_grad_enabled(False)
    cfg=None
    if args.network == "mobile0.25":
        cfg=cfg_mnet
    elif args.network == "resnet50":
        cfg=cfg_re50
    # net and model
    net=RetinaFace(cfg=cfg, phase='test')
    net=load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    cudnn.benchmark=True
    device=torch.device("cpu" if args.cpu else "cuda")
    net=net.to(device)

    #对train和test文件夹下面的所有文件都进行人脸检测，对齐和剪裁
    for i in range(0, 8):
        root=os.path.join(split_Dir,str(i))
        transform_image(root)
    for i in range(0, 8):
        root=os.path.join(test_Dir,str(i))
        transform_image(root)
