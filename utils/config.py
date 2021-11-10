import os


class Config:
    train_img_path = os.path.abspath(
        '/home/yy/project/s2anet/data/hjj/train/images_png')  
    train_gt_path = os.path.abspath(
        '/home/yy/project/s2anet/data/hjj/train/labels')

    pths_path = './data/pths'
    preTrained = '/home/ftpuser/ftp/upload/model/resnet50-19c8e357.pth'
    # checkPoint = './data/pths/model_epoch_20.pth'
    # logDir = './data/log/'
    batch_size = 32
    lr = 1e-3
    num_workers = 16
    epoch_iter = 120
    save_interval = 5
    length = 512
    locAnchor = [0, 16, 32, 64, 128, 256, 512]

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # categories = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
    #               'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']

    # categories = ['aircraft carrier', 'warcraft', 'merchant ship']
    categories = ['1', '2', '3', '4', '5']
config = Config()
