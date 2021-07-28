from __future__ import print_function
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data import cfg_mnet, cfg_re50
from layers.functions.prior_box import PriorBox
from utils.nms.py_cpu_nms import py_cpu_nms
from utils.datasets import LoadImages
from utils.general import increment_path
from pathlib import Path
import cv2
from models.retinaface import RetinaFace
from utils.box_utils import decode, decode_landm
import time
from PIL import Image, ImageDraw
import math
from collections import defaultdict
import face_recognition  # install from https://github.com/ageitgey/face_recognition
import torchvision.models as models



parser = argparse.ArgumentParser(description='Retinaface')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
parser.add_argument('--project', default='runs/face_recognition', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--name', default='exp', help='save results to project/name')
args = parser.parse_args()


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

def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array)
    draw = ImageDraw.Draw(origin_img)
    for facial_feature in landmarks.keys():
        draw.point(landmarks[facial_feature])
    origin_img.show()

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

    pil_img = Image.fromarray(image_array)
    left, top, right, bottom = [int(i) for i in [left, top, right, bottom]]
    cropped_img = pil_img.crop((left, top, right, bottom))
    cropped_img = np.array(cropped_img)
    return cropped_img, left, top

def transfer_landmark(landmarks, left, top):
    """transfer landmarks to fit the cropped face
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :param left: left coordinates of cropping
    :param top: top coordinates of cropping
    :return: transferred_landmarks with the same structure with landmarks, but different values
    """
    transferred_landmarks = defaultdict(list)
    for facial_feature in landmarks.keys():
        for landmark in landmarks[facial_feature]:
            transferred_landmark = (landmark[0] - left, landmark[1] - top)
            transferred_landmarks[facial_feature].append(transferred_landmark)
    return transferred_landmarks


def get_labels(dataset_name):
    if dataset_name == 'fer2013':
        return {0:'angry', 1:'disgusted', 2:'fearful', 3:'happy', 4:'sad', 5:'surprised', 6:'neutral'}
    elif dataset_name == 'imdb':
        return {0: 'woman', 1: 'man'}
    elif dataset_name == 'KDEF':
        return {0: 'AN', 1: 'DI', 2: 'AF', 3: 'HA', 4: 'SA', 5: 'SU', 6: 'NE'}
    else:
        raise Exception('Invalid dataset name')

def draw_text(coordinates, image_array, text, color, x_offset=0, y_offset=0,
                                                font_scale=2, thickness=2):
    x, y = coordinates[:2]
    cv2.putText(image_array, text, (x + x_offset, y + y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, thickness, cv2.LINE_AA)

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    cfg = None

    if args.network == "mobile0.25":
        cfg = cfg_mnet
    elif args.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args.trained_model, args.cpu)
    net.eval()
    print('Finished loading model!')
    #print(net)
    cudnn.benchmark = True
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device=torch.device("cpu")
    net = net.to(device)

    vid_path, vid_writer=None, None
    save_txt=args.save_txt
    save_image=args.save_image
    save_dir=increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    resize = 1

    # testing begin

    # imgsz = args.source
    # imgsz = check_img_size(imgsz)  # check img_size
    filepath=args.source
    print("filepath", filepath)
    dataset=LoadImages(filepath)

    yui_image=face_recognition.load_image_file("./data/known_faces/aragaki yui.jpeg")
    yui_face_encoding=face_recognition.face_encodings(yui_image)[0]

    obama_image=face_recognition.load_image_file("./data/known_faces/obama.jpg")
    obama_face_encoding=face_recognition.face_encodings(obama_image)[0]

    known_faces=[
        yui_face_encoding,
        obama_face_encoding
    ]
    emotion_labels=get_labels('fer2013')
    emotionnet=models.resnext50_32x4d().to(device)
    checkpoint = torch.load('./models/trained_models/resnet50_0721_94%/test_model0721.ckpt')
    emotionnet.load_state_dict(checkpoint)

    for path, imag, im0s, vid_cap in dataset:
        frame=getattr(dataset, 'frame', 0)
        p=path
        p=Path(p)  # to Path
        save_path=str(save_dir / p.name)  # img.jpg
        txt_path=str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt

        img=np.float32(im0s)
        im_height, im_width, _=img.shape
        scale=torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img-=(104, 117, 123)
        img=img.transpose(2, 0, 1)
        img=torch.from_numpy(img).unsqueeze(0)
        img=img.to(device)
        scale=scale.to(device)

        tic=time.time()
        loc, conf, landms=net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic)        )

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
        im0=im0s.copy()
        face_names=[]
        if args.save_image:
            for b in dets:
                text = "{:.4f}".format(b[4])
                b = list(map(int, b))
                cv2.rectangle(im0, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
                cx = b[0]
                cy = b[1] + 12
                cv2.putText(im0, text, (cx, cy),
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))

                face_landmarks_dict={'left_eye':[(b[5], b[6])],
                                     'right_eye':[(b[7], b[8])],
                                     'nose':[(b[9], b[10])],
                                     'left_mouth':[(b[11], b[12])],
                                     'right_mouth':[(b[13], b[14])],}
                aligned_face, eye_center, angle=align_face(image_array=im0, landmarks=face_landmarks_dict)
                rotated_landmarks=rotate_landmarks(landmarks=face_landmarks_dict,
                                                   eye_center=eye_center, angle=angle, row=im0.shape[0])
                cropped_face, left, top=corp_face(image_array=aligned_face,landmarks=rotated_landmarks)

                try:
                    img=cropped_face[..., ::-1]
                    unknown_encoding=face_recognition.face_encodings(img)[0]
                    match=face_recognition.compare_faces(known_faces, unknown_encoding, tolerance=0.50)
                    name=None
                    if match[0]:
                        name="aragaki yui"
                    elif match[1]:
                        name="obama"
                    face_names.append(name)

                    # Draw a box around the face
                    cv2.rectangle(im0, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(im0, (b[0], b[3] - 15), (b[2], b[3]), (0, 0, 255), cv2.FILLED)

                    font=cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(im0, name, (b[0] + 6, b[3] - 6), font, 0.5, (255, 255, 255), 1)

                    # emotion classification
                    cropped_face=Image.fromarray(img)
                    cropped_face=cropped_face.resize((256, 256), Image.ANTIALIAS)
                    cropped_face=np.array(cropped_face).astype(np.uint8)
                    cropped_face=cropped_face[np.newaxis, :, :, :]
                    cropped_face=cropped_face.transpose(0, 3, 1, 2)
                    cropped_face=torch.from_numpy(cropped_face)
                    cropped_face=cropped_face.to(device)
                    cropped_face=cropped_face.float()
                    emotion=emotionnet(cropped_face)
                    emotion=emotion.cuda().data.cpu().numpy()
                    emotion_label_arg=np.argmax(emotion)
                    emotion_text=emotion_labels[emotion_label_arg]
                    cv2.putText(im0, emotion_text, (b[0] + 12, b[3] + 12), font, 0.5, (255, 255, 255), 1)
                except IndexError as e:
                    print('IndexError:', e)

                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path=save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps=vid_cap.get(cv2.CAP_PROP_FPS)
                            w=int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h=int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer=cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)




