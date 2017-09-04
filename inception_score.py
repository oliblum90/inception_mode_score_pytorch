import torchvision
from torchvision.models.inception import model_urls
import numpy as np
import scipy.misc
from torchvision import models, transforms
from PIL import Image
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import tqdm


class InceptionScore:
    
    def __init__(self, gpu = True):
        """
        Class for the computation of the inception score
        
        Parameters
        ----------
        gpu : bool
            if True:  computation is executed on the gpu
            if False: computation is executed on the cpu
        """
        
        # set global vars
        self.gpu = gpu
        
        # load model
        try:
            self.incept = torchvision.models.inception_v3(pretrained=True)
        except:
            name = 'inception_v3_google'
            model_urls[name] = model_urls[name].replace('https://', 'http://')
            self.incept = torchvision.models.inception_v3(pretrained=True)
        self.incept.training = False
        self.incept.transform_input = False
        if self.gpu:
            self.incept = self.incept.cuda()
        self.incept.eval()
    
        # init data transformer
        normalize = transforms.Normalize(
           mean=[0.485, 0.456, 0.406],
           std=[0.229, 0.224, 0.225]
        )
        self.preprocess = transforms.Compose([
           transforms.ToTensor(),
           normalize
        ])
        
    
    def score(self, imgs, batch_size=32, splits=10):
        """
        Function to compute the inception score
        
        Parameters
        ----------
        imgs : numpy array
            array of the shape (N, X, Y, C)
        batch_size : int
            batch size for the prediction with the inception net
        splits : int
            The inception score is computed for a package of images.
            The variable 'splits' defines the number of these packages.
            Multiple computations of the score (for each package one) are 
            needed to compute a standard diviation (error) for the final
            score.
        """
        
        # preprocess images
        if imgs.shape[0] != 299 or imgs.shape[1] != 299:
            imgs = np.array([scipy.misc.imresize(img, (299, 299)) for img in imgs])
        n_batches = 1 + (len(imgs) / batch_size)
        batches = np.array_split(imgs, n_batches)
        
        # get prediction vectors of inception net for images
        preds = []
        for batch in tqdm(batches):
            imgs = [Image.fromarray(img) for img in batch]
            imgs = torch.stack([self.preprocess(img) for img in imgs])
            if self.gpu:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            pred = self.incept(imgs)
            pred = F.softmax(pred)
            preds.append(pred.data.cpu().numpy())    
        preds = np.concatenate(preds)
        
        # compute inception score
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits): \
                         ((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
            
        return np.mean(scores), np.std(scores)
    
    
if __name__ == "__main__":
    
    trainset = torchvision.datasets.CIFAR10(root = '/tmp', download=True)
    x = trainset.train_data
    incept_score = InceptionScore()
    mean, std = incept_score.score(x)
    
    print "score = {} +- {}".format(mean, std)
    