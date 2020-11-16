from flask import Flask, request, Response, jsonify
import jsonpickle
import numpy as np
from PIL import Image
import cv2
import os
import multiprocessing
import random
import os, sys
import json

import torch
import torchvision
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.parallel
import torch.nn.functional as F

import volume_net.modules as modules
import volume_net.net as net
import volume_net.resnet as  resnet
import volume_net.densenet as densenet
import volume_net.senet as senet
from volume_net.demo_transform import *
from volume_net.volume import *

import matplotlib.image
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import PIL.ExifTags
from ast import literal_eval

from shapely.ops import cascaded_union, unary_union
from shapely.geometry import Polygon

from flasgger import Swagger
from werkzeug.utils import secure_filename
from flasgger.utils import swag_from

from detectron2.structures import BoxMode
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode

app = Flask(__name__)
swagger = Swagger(app)


def define_model(is_resnet, is_densenet, is_senet):
    if is_resnet:
        original_model = resnet.resnet50(pretrained = True)
        Encoder = modules.E_resnet(original_model) 
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])
    if is_densenet:
        original_model = densenet.densenet161(pretrained=True)
        Encoder = modules.E_densenet(original_model)
        model = net.model(Encoder, num_features=2208, block_channel = [192, 384, 1056, 2208])
    if is_senet:
        original_model = senet.senet154(pretrained='imagenet')
        Encoder = modules.E_senet(original_model)
        model = net.model(Encoder, num_features=2048, block_channel = [256, 512, 1024, 2048])

    return model


def mask_to_polygons(mask):
    mask = np.ascontiguousarray(mask)
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]
    return res, has_holes


def convert_masks_annotations(points):
    points_x = points[::2]
    points_y = points[1::2]

    anno_format = []
    for i in range(points_x.shape[0]):
        anno_format.append([int(points_x[i]), int(points_y[i])])

    return np.array(anno_format)


def get_plate_mask(masks):

    polys = []

    x_maxi = 0
    x_mini = sys.maxsize
    y_maxi = 0
    y_mini = sys.maxsize
    for mask in masks:

        points_x = mask[::2]
        points_y = mask[1::2]

        anno_format = []
        for i in range(points_x.shape[0]):
            anno_format.append((points_x[i], points_y[i]))

            if points_x[i]>x_maxi:
                x_maxi = points_x[i]
            if points_y[i]>y_maxi:
                y_maxi = points_y[i]
            if points_x[i]<x_mini:
                x_mini = points_x[i]
            if points_y[i]<y_mini:
                y_mini = points_y[i]

        polys.append(Polygon(anno_format))

    try:
        x, y = cascaded_union(polys).exterior.coords.xy
    except:
        x = [x_mini, x_maxi, x_maxi, x_mini]
        y = [y_mini, y_mini, y_maxi, y_maxi]

    anno_format = []
    for i in range(len(x)):
        anno_format.append([int(x[i]), int(y[i])])
        
    #boundary = gpd.GeoSeries(cascaded_union(polys))
    #boundary.plot(color = 'red')
    #plt.show()

    return anno_format


def predict_food(img):

    height, width = img.shape[:2]

    try:
        exif = {PIL.ExifTags.TAGS[k]: v for k, v in Image.open(path_img)._getexif().items() if k in PIL.ExifTags.TAGS}
        f1, f2 = exif["FocalLength"]
        focal_length = float(f1)/float(f2)
    except:
        focal_length = 5.0

    #print("Focal Length: " + str(focal_length) + " mm")


    # Predict depth
    transform = transforms.Compose([Scale([320, 240]), CenterCrop([304, 228]), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)

    image = transform(im_pil)
    with torch.no_grad():
        if gpu:
            image = image.unsqueeze(0).cuda()
        else:
            image.unsqueeze(0)
        out = model_depth(image)
        out = out.view(out.size(2),out.size(3)).data.cpu().numpy()
        max_pix = out.max()
        min_pix = out.min()
        out = (out-min_pix)/(max_pix-min_pix)*255
        out = cv2.resize(out,(width,height),interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        out3 = cv2.merge((out, out, out))
        #cv2.imshow("Grey depth", out3)
        #out_color = cv2.applyColorMap(out, cv2.COLORMAP_JET)


    # Predict masks and bboxes
    outputs = predictor(img)
    #v = Visualizer(img[:, :, ::-1], metadata=food_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("Img masks", v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)

    predictions = outputs["instances"].to("cpu")
    bboxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    bboxes = bboxes.tensor.numpy()
    scores = predictions.scores if predictions.has("scores") else None
    scores = scores.numpy()
    masks = np.asarray(predictions.pred_masks)
    masks = [mask_to_polygons(x)[0][0] for x in masks]
    #print(bboxes, scores, masks)

    # Get complete plate mask
    plate_mask = get_plate_mask(masks)
    json_final = {"ingredients": [], "probabilities": [], "volumes": []}
    volumes = []
    for i, box in enumerate(bboxes):

        mask_pp_format = convert_masks_annotations(masks[i])

        img_rec = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        img_rec = cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rec)
        input_img = valid_tfms(im_pil)

        if gpu:
            input_img = input_img.unsqueeze(0).cuda() 
        else:
            input_img = input_img.unsqueeze(0).cpu()

        var_ip = Variable(input_img)
        output = F.softmax(model_rec(var_ip)[0], dim=0)

        #_, pred = torch.max(output)

        indexes_top5 = output.cpu().detach().numpy().argsort()[-5:][::-1]
        probs_top5 = np.take(output.cpu().detach().numpy(), indexes_top5).tolist()
        #food_rec = labels[int(pred[0])]

        json_data = {"shapes": [{"label": "plate", "points": plate_mask}, {"label": "ingredient", "points": mask_pp_format}]}
        volume = get_volume(out, json_data)["ingredient"]
        volumes.append(volume)

        labels_top5 = []
        for index in indexes_top5:
            labels_top5.append(labels[int(index)])

        json_final["probabilities"].append(probs_top5)
        json_final["ingredients"].append(labels_top5)

    json_final["volumes"] = volumes

    #json_vol = get_volume(out, json_data)

    return json_final


def predict_food(img):

    height, width = img.shape[:2]

    try:
        exif = {PIL.ExifTags.TAGS[k]: v for k, v in Image.open(path_img)._getexif().items() if k in PIL.ExifTags.TAGS}
        f1, f2 = exif["FocalLength"]
        focal_length = float(f1)/float(f2)
    except:
        focal_length = 5.0

    #print("Focal Length: " + str(focal_length) + " mm")


    # Predict depth
    transform = transforms.Compose([Scale([320, 240]), CenterCrop([304, 228]), ToTensor(), Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(image)

    image = transform(im_pil)
    with torch.no_grad():
        image = image.unsqueeze(0)
        if gpu:
            image = image.cuda()
        out = model_depth(image)
        out = out.view(out.size(2),out.size(3)).data.cpu().numpy()
        max_pix = out.max()
        min_pix = out.min()
        out = (out-min_pix)/(max_pix-min_pix)*255
        out = cv2.resize(out,(width,height),interpolation=cv2.INTER_CUBIC).astype(np.uint8)
        out3 = cv2.merge((out, out, out))
        #cv2.imshow("Grey depth", out3)
        #out_color = cv2.applyColorMap(out, cv2.COLORMAP_JET)


    # Predict masks and bboxes
    outputs = predictor(img)
    #v = Visualizer(img[:, :, ::-1], metadata=food_metadata, scale=0.8, instance_mode=ColorMode.IMAGE_BW)
    #v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    #cv2.imshow("Img masks", v.get_image()[:, :, ::-1])
    #cv2.waitKey(0)

    predictions = outputs["instances"].to("cpu")
    bboxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
    bboxes = bboxes.tensor.numpy()
    scores = predictions.scores if predictions.has("scores") else None
    scores = scores.numpy()
    masks = np.asarray(predictions.pred_masks)
    masks = [mask_to_polygons(x)[0][0] for x in masks]
    #print(bboxes, scores, masks)

    # Get complete plate mask
    plate_mask = get_plate_mask(masks)
    json_final = {"ingredients": [], "probabilities": [], "volumes": []}
    volumes = []
    for i, box in enumerate(bboxes):

        mask_pp_format = convert_masks_annotations(masks[i])

        img_rec = img[int(box[1]):int(box[3]), int(box[0]):int(box[2])]

        img_rec = cv2.cvtColor(img_rec, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(img_rec)
        input_img = valid_tfms(im_pil)

        if gpu:
            input_img = input_img.unsqueeze(0).cuda() 
        else :
            input_img = input_img.unsqueeze(0)

        var_ip = Variable(input_img)
        output = F.softmax(model_rec(var_ip)[0], dim=0)

        #_, pred = torch.max(output)

        indexes_top5 = output.cpu().detach().numpy().argsort()[-5:][::-1]
        probs_top5 = np.take(output.cpu().detach().numpy(), indexes_top5).tolist()
        #food_rec = labels[int(pred[0])]

        json_data = {"shapes": [{"label": "plate", "points": plate_mask}, {"label": "ingredient", "points": mask_pp_format}]}
        volume = get_volume(out, json_data)["ingredient"]
        volumes.append(volume)

        labels_top5 = []
        for index in indexes_top5:
            labels_top5.append(labels[int(index)])

        json_final["probabilities"].append(probs_top5)
        json_final["ingredients"].append(labels_top5)

    json_final["volumes"] = volumes

    #json_vol = get_volume(out, json_data)

    return json_final


# Load recognition network
f = open("labels.txt", "r")
labels = f.read().split("\n")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = True if torch.cuda.is_available() else False

model_rec = models.wide_resnet101_2(pretrained=True) # model
num_ftrs = model_rec.fc.in_features
model_rec.fc = nn.Linear(in_features=num_ftrs, out_features=101, bias=True)
model_rec.load_state_dict(torch.load("model_food_35.pth", map_location=torch.device('cpu'))) 
if gpu:
    model_rec = model_rec.cuda()
model_rec.eval()

imagenet_stats = [(0.485, 0.456, 0.406), (0.229, 0.224, 0.225)]
valid_tfms = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(imagenet_stats[0], imagenet_stats[1])])

 # Load detection masks network
food_metadata = MetadataCatalog.get("foodSeg/train").set(thing_classes=["foodSeg"])
cfg = get_cfg()
cfg.merge_from_file("detectron2_repo/configs/COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.MODEL.WEIGHTS = "model_masks_5000.pth"
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TEST = ("foodSeg/val", )
if not gpu:
    cfg.MODEL.DEVICE = "cpu"
predictor = DefaultPredictor(cfg)

# Loas depth network
model_depth = define_model(is_resnet=False, is_densenet=False, is_senet=True)
model_depth = torch.nn.DataParallel(model_depth)
model_depth.load_state_dict(torch.load('volume_net/pretrained_model/model_senet', map_location=torch.device('cpu')))
if gpu:
    model_depth = torch.nn.DataParallel(model_depth).cuda()
model_depth.eval()