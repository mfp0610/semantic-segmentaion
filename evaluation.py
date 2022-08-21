import numpy as np
import torch
import cv2

def cal_miou(img_pre, img_gt, isunetpp = False) :
    img_pre = torch.argmax(img_pre, 1)
    bs, w, h = img_pre.shape
    miou = 0
    for i in range(bs):
        pred, mask = img_pre[i], img_gt[i]
        union = torch.logical_or(pred, mask).sum()
        inter = ((pred + mask) == 2).sum()
        if union < 1e-5:
            return 0
        miou += inter / union
        #print(i, miou)
    return miou / bs


def get_boundary(img, is_mask) :
    if not is_mask:
        img = torch.argmax(img, 1).cpu().numpy().astype('float64')
    else:
        img = img.cpu().numpy()
    bs, w, h = img.shape
    new_img = np.zeros([bs, w + 2, h + 2])
    mask_erode = np.zeros([bs, w, h])
    dil = int(round(0.02 * np.sqrt(w ** 2 + h ** 2)))
    if dil < 1:
        dil = 1
    for i in range(bs):
        new_img[i] = cv2.copyMakeBorder(img[i], 1, 1, 1, 1, \
            cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    for i in range(bs):
        img_erode = cv2.erode(new_img[i], kernel, iterations = dil)
        mask_erode[i] = img_erode[1: w + 1, 1: h + 1]
    return torch.from_numpy(img - mask_erode)


def cal_biou(img_pre, img_gt) :
    img_pre = get_boundary(img_pre, is_mask=False)
    img_gt = get_boundary(img_gt, is_mask=True)
    bs, w, h = img_pre.shape
    inter, union = 0, 0
    for i in range(bs):
        pred, mask = img_pre[i], img_gt[i]
        inter += ((pred * mask) > 0).sum()
        union += ((pred + mask) > 0).sum()
    if union < 1:
        return 0
    biou = inter / union
    return biou


def cal_miou_pp(img_pre, img_gt):
    img_pre = img_pre.round().squeeze(1)
    bs, w, h = img_pre.shape
    miou = 0
    for i in range(bs):
        pred, mask = img_pre[i], img_gt[i]
        union = torch.logical_or(pred, mask).sum()
        inter = ((pred + mask) == 2).sum()
        if union < 1e-5:
            return 0
        miou += inter / union
    return miou / bs


def get_boundary_pp(pic, is_mask):
    if not is_mask:
        pic = pic.round().squeeze(1).cpu().numpy().astype('float64')
    else:
        pic = pic.cpu().numpy()
    bs, w, h = pic.shape
    new_pic = np.zeros([bs, w + 2, h + 2])
    mask_erode = np.zeros([bs, w, h])
    dil = int(round(0.02 * np.sqrt(w ** 2 + h ** 2)))
    if dil < 1:
        dil = 1
    for i in range(bs):
        new_pic[i] = cv2.copyMakeBorder(pic[i], 1, 1, 1, 1, \
            cv2.BORDER_CONSTANT, value = 0)
    kernel = np.ones((3, 3), dtype = np.uint8)
    for i in range(bs):
        pic_erode = cv2.erode(new_pic[i], kernel, iterations = dil)
        mask_erode[i] = pic_erode[1: w + 1, 1: h + 1]
    return torch.from_numpy(pic - mask_erode)


def cal_biou_pp(img_pre, img_gt):
    img_pre = get_boundary_pp(img_pre, is_mask=False)
    img_gt = get_boundary_pp(img_gt, is_mask=True)
    bs, w, h = img_pre.shape
    inter, union = 0, 0
    for i in range(bs):
        pred, mask = img_pre[i], img_gt[i]
        inter += ((pred * mask) > 0).sum()
        union += ((pred + mask) > 0).sum()
    if union < 1:
        return 0
    biou = inter / union
    return biou
