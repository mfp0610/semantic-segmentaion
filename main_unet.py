import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import RMSprop
import cv2
from tensorboardX import SummaryWriter

from DataLoader import WHDataset
import DataLoader as DL
import config.unet_config as settings
from demo.UNet import UNet
import evaluation as eva

if __name__ == "__main__":
    # Load Dataset
    root_dir = "./dataset/weizmann_horse_db"
    horse_dir, mask_dir = "horse", "mask"
    horse_dataset = WHDataset(root_dir, horse_dir, mask_dir)
    train_loader, test_loader = DL.WHDataLoader(horse_dataset, 0.85, \
        batch_size = 8, num_workers = 8)

    # Training Configuration
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epochs, lr, decay, momen = \
        settings.EPOCH, settings.LR, settings.WEIGHT_DECAY, settings.MOMENTUM
    
    # model configuration
    #  define the model
    model = UNet(3,2).to(device)
    #  define the loss
    loss = nn.CrossEntropyLoss()
    #  define the optimizer 
    optimizer = RMSprop(model.parameters(), lr = lr, weight_decay = decay, momentum = momen)
    # logger
    writer = SummaryWriter(settings.LOG_DIR + "tb/exp_unet/")
    
    # training and testing
    bmiou, bbiou, bepoch= 0., 0., 0
    for epoch in range(epochs):
        tloss, tmiou, tbiou = [], [], []
        # train
        for index, (image, label) in enumerate(train_loader, 1):
            torch.cuda.empty_cache()
            image, label=image.to(device), label.to(device)
            optimizer.zero_grad()
            pred = model(image)
            iloss = loss(pred, label)
            tloss.append(iloss) # record the loss
            iloss.backward()
            optimizer.step()

        # test and record
        tloss=sum(tloss)
        with torch.no_grad():
            for timage, tlabel in test_loader:
                timage, tlabel = timage.to(device), tlabel.to(device)
                tpred = model(timage)
                tmiou.append(eva.cal_miou(tpred, tlabel).cpu().detach())
                tbiou.append(eva.cal_biou(tpred, tlabel).cpu().detach())
        miou = sum(tmiou).item()/len(tmiou)
        biou = sum(tbiou).item()/len(tbiou)
        
        # writer to tensorboardX
        writer.add_scalar('loss', tloss.item(), global_step = epoch)
        writer.add_scalar('miou', sum(tmiou).item()/len(tmiou), global_step = epoch)
        writer.add_scalar('biou', sum(tbiou).item()/len(tbiou), global_step = epoch)
    
        # save the model
        if epoch % 50 == 0:
            torch.save(model.state_dict(), \
                settings.MODEL_DIR + "unet/unet_epoch"+str(epoch) + ".pth")
        if miou >= bmiou and biou >= bbiou:
            torch.save(model.state_dict(), \
                settings.MODEL_DIR + "unet/unet_bestmodel.pth")
            bmiou, bbiou, bepoch = miou, biou, epoch
        if epoch % 10 == 0:
            print('epoch:',epoch+1, ' loss:',tloss.item(), ' miou:',miou, ' biou:',biou)
    
    # best performance
    print("======== best performance ========")
    print('epoch:',bepoch+1, ' miou:',bmiou, ' biou:',bbiou)
