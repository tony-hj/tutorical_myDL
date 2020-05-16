from torchvision import transforms

input_size = 380

mean = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2470, 0.2435, 0.2616])

trans_tta_1 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_2 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=Config.mean, std=Config.std)
])
trans_tta_3 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomRotation((-25, 25)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_4 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomHorizontalFlip(0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_5 = transforms.Compose([
    transforms.Resize(input_size),
    transforms.RandomVerticalFlip(0.25),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
trans_tta_6 = transforms.Compose([
    Resize_propotion(input_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

tta_trans_list = [trans_tta_1,trans_tta_2,trans_tta_3,trans_tta_4,trans_tta_5,trans_tta_6]
    
test_transform = transforms.Compose([
    torchvision.transforms.Resize(config.input_size),
    torchvision.transforms.CenterCrop(config.input_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(config.mean, config.std),
])

train_transform = transforms.Compose([
    torchvision.transforms.Resize(config.input_size),
    torchvision.transforms.CenterCrop(config.input_size),
    torchvision.transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(config.mean, config.std),
])

batch_size = 20
num_workers = 0
num_classes = 2
root = ''
epochs = 20
model_path = ''
out_dir = './output'
label_smooth = True
LR = 1e-3




