import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

class EvaluateFrameEstimation():
    def __init__(self,batch_size,test_dl,model,pretrainedModel,device,saveDir,maskThreshold=60,graphYmax=150, debug=False):
        self.test_dl = test_dl # evaluation img data
        self.maskThreshold = maskThreshold
        self.saveDir=saveDir
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)
        self.debug = debug
        self.graphYmax = graphYmax #grapg range in y
        self.batch_size=batch_size

        #inference
        model.load_state_dict(torch.load(pretrainedModel, map_location=device))
        model.eval()
        x = next(iter(test_dl))
        with torch.no_grad():
            rec_img, mask = model(x.to(device))

        self.x, self.rec_img, self.mask = x.to("cpu"), rec_img.to("cpu"), mask.to("cpu")

        #evaluation
        self.evaluation()

    def calculateDifference(self,img):
        img = np.array(img).flatten() #flatten img
        countIteration = 0
        print(img.shape)
        mean = round(img.mean(),1) #mean
        std = round(np.std(img),1) #standard deviation
        return mean, std

    def evaluation(self):
        # マスクしていた部分は元の画像を用いる
        imgs = self.rec_img * self.mask + self.x * (1 - self.mask)
        imgs = np.array(255*((imgs.data + 1) / 2),np.uint8) #0~255
        imgs_original = np.array(255*((self.x.data + 1) / 2),np.uint8) #original image
        i = 1 #iteraion count
        Means = [] #mean difference for estimation accuracy
        Stds = [] #standard deviation for estimation accuracy
        counter = 0
        for img,original in zip(imgs[:self.batch_size],imgs_original[:self.batch_size]):
            #img  = np.expand_dims(img,0)
            #print(img.shape)
            #print(np.transpose(torch.squeeze(img).numpy()).shape)
            # 出力が線形変換のため0-1になっているとは限らないためclipする
            #print(img)
            #img  = np.expand_dims(img,0)
            #print(img.shape)
            #print(np.transpose(torch.squeeze(img).numpy()).shape)
            # 出力が線形変換のため0-1になっているとは限らないためclipする
            #img = np.clip(np.transpose(img, (1,2,0)), 0, 1)#torch.squeeze(img).numpy(), (1,2,0)), 0, 1) #(C,H,W) -> (H,W,C)
            #exclude non laser dateset
            if counter!= 9 or counter!=10 or counter!=15 or counter!=21 or counter!=25 or counter!=26 or counter!=28:
                original = original[:,162:,162:] #crop img
                img = img[:,162:,162:]
                original = np.transpose(original,(1,2,0))
                img = np.transpose(img,(1,2,0))
                ret,threshMask = cv2.threshold(original,self.maskThreshold,255,cv2.THRESH_BINARY) #img element should be unsigned int8 or float
                #find contours
                contours, hierarchy = cv2.findContours(np.array(threshMask,dtype=np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


                Area = []
                #detect laser area
                if (bool(contours)==True):
                    #面積(px*px)
                    countContour=1
                    displayImg = original.copy()
                    for j in contours:
                        area = cv2.contourArea(j)
                        Area.append(area)
                        countContour+=1
                        if self.debug:
                            cv2.drawContours(displayImg,j,contourIdx=-1,color=(20*countContour,0,0),thickness=2)
                            cv2.imwrite(f"contours_{i}.jpg",displayImg)

                    if Area:
                        ind = np.argmax(Area)
                        cnt = contours[ind]

                    maxArea = np.max(Area)
                    print(maxArea)
                    dif = abs(img - original)
                    cv2.bitwise_and(dif, threshMask)
                    mean,std = self.calculateDifference(dif)
                    Means.append(mean)
                    Stds.append(std)
                    print(f"{i}'s imgs mean difference : {mean}, std : {std}\n")
                    if self.debug:
                        cv2.imwrite(os.path.join(self.saveDir,f"{i}_dif.jpg"),dif)
                    i+=1

        Means = np.array(Means)
        Stds = np.array(Stds)
        averageDiff = round(Means.mean(),1)
        averageStd = round(np.mean(Stds),1)

        fig,ax = plt.subplots(2,1,figsize=(10,10))
        ax[0].plot(Means,label=f"mean difference = {averageDiff}")
        ax[0].set_ylim(0,self.graphYmax)
        ax[0].legend()
        ax[1].plot(Stds,label=f"mead std = {averageStd}")
        ax[1].set_ylim(0,self.graphYmax)
        ax[1].legend()
        fig.show()
        fig.savefig(os.path.join(self.saveDir,"estimationEvaluation.jpg"))


        print(f"average difference : {averageDiff}, average std : {averageStd}\n")