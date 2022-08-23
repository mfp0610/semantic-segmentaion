import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.autograd import Variable
import cv2
from tensorboardX import SummaryWriter

from DataLoader import WHDataset
import DataLoader as DL
import config.unetpp_config as settings
from demo.UNetPP import UNETPP
import demo.UNetPP as unetpp
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
    epochs, lr, beta_min, beta_max, eps, decay = \
        settings.EPOCH, settings.LR, settings.BETA_MIN, settings.BETA_MAX, \
        settings.EPS, settings.WEIGHT_DECAY
    
    # model configuration
    #  define the model
    model = UNETPP(3,1).to(device)
    #  define the loss 
     # loss = nn.CrossEntropyLoss()
    #  define the optimizer 
    optimizer = Adam(model.parameters(), lr = lr, betas = (beta_min, beta_max), \
        eps = eps, weight_decay = decay)
    # logger
    writer = SummaryWriter(settings.LOG_DIR + "tb/exp_unetpp/")

    ite_num = 0
    #iloss = 0.0
    #itar_loss = 0.0
    #ite_num4val = 0
    #save_frq = 50 # save the model every 2000 iterations

    miou,biou=[],[]
    tmiou,tbiou=[],[]
    
    # training and testing
    bmiou, bbiou, bepoch= 0., 0., 0
    for epoch in range(epochs):
        iloss = 0.0
        itar_loss = 0.0
        ite_num4val = 0
        for i, data in enumerate(train_loader):
            ite_num += 1
            ite_num4val += 1
            inputs, labels = data
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(torch.FloatTensor)
            # wrap them in Variable
            inputs_v, labels_v = Variable(inputs.to(device), requires_grad = False), \
                Variable(labels.to(device), requires_grad = False)
            optimizer.zero_grad()
            d0, d1, d2, d3, d4, d5, d6 = model(inputs_v)
            loss1, loss = unetpp.multi_bce_loss_fusion(d0, d1, d2, d3, d4, d5, d6, labels_v)
            loss.backward()
            optimizer.step()
            iloss += loss.item()
            itar_loss += loss1.item()
            del d0, d1, d2, d3, d4, d5, d6, loss1, loss

        with torch.no_grad():
            for timage, tlabel in test_loader:
                timage, tlabel = timage.to(device), tlabel.to(device)
                tpred = model(timage)
                tmiou.append(eva.cal_miou_pp(tpred[0], tlabel.squeeze(0)).cpu().detach())
                tbiou.append(eva.cal_biou_pp(tpred[0], tlabel.squeeze(0)).cpu().detach())
        miou = sum(tmiou).item()/len(tmiou)
        biou = sum(tbiou).item()/len(tbiou)
        
        # writer to tensorboardX
        writer.add_scalar('loss', iloss/ite_num4val, global_step = epoch)
        writer.add_scalar('miou', sum(tmiou).item()/len(tmiou), global_step = epoch)
        writer.add_scalar('biou', sum(tbiou).item()/len(tbiou), global_step = epoch)
    
        # save the model
        if epoch % 50 == 0:
            torch.save(model.state_dict(), settings.MODEL_DIR+"unetpp/unetpp" + str(epoch) + ".pth")
        if miou >= bmiou and biou >= bbiou:
            torch.save(model.state_dict(), \
                settings.MODEL_DIR + "unetpp/unetpp_bestmodel.pth")
            bmiou, bbiou, bepoch = miou, biou, epoch
        if epoch % 10 == 0:
            print('epoch:',epoch+1, ' train loss:',iloss/ite_num4val,  ' tar:',itar_loss/ite_num4val, \
                ' miou:',sum(tmiou).item()/len(tmiou), ' biou:', sum(tbiou).item()/len(tbiou))

    # best performance
    print("======== best performance ========")
    print('epoch:',bepoch+1, ' miou:',bmiou, ' biou:',bbiou)
