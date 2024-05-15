import os
import cv2
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
import albumentations
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

from tqdm import tqdm
import random


class MelanomaDataset(Dataset):
    def __init__(self, csv, mode, meta_features, transform=None):

        self.csv = csv.reset_index(drop=True)
        self.mode = mode
        self.use_meta = meta_features is not None
        self.meta_features = meta_features
        self.transform = transform

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):

        row = self.csv.iloc[index]

        image = cv2.imread(row.filepath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        

        if self.transform is not None:
        	res = self.transform(image=image)
        	image = res['image'].astype(np.float32)
        else :
            image = image.astype(np.float32)

        image = image.transpose(2, 0, 1)

        if self.use_meta:
            data = (torch.tensor(image).float(), torch.tensor(self.csv.iloc[index][self.meta_features]).float())
        else:
            data = torch.tensor(image).float()

        if self.mode == 'test':
            return data, torch.tensor(self.csv.iloc[index].target).long()
        else:
            return data, torch.tensor(self.csv.iloc[index].target).long()

class DrawHair():
    """
    Draw a random number of pseudo hairs

    Args:
        hairs (int): maximum number of hairs to draw
        width (tuple): possible width of the hair in pixels
    """

    def __init__(self, hairs:int = 4, width:tuple = (1, 2)):
        self.hairs = hairs
        self.width = width

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to draw hairs on.

        Returns:
            PIL Image: Image with drawn hairs.
        """
        if not self.hairs:
            return img
        
        height, width = img.shape[:2]
        
        for _ in range(random.randint(0, self.hairs)):
            # The origin point of the line will always be at the top half of the image
            origin = (random.randint(0, width), random.randint(0, height // 2))
            # The end of the line 
            end = (random.randint(0, width), random.randint(0, height))
            color = (0, 0, 0)  # color of the hair. Black.
            cv2.line(img, origin, end, color, random.randint(self.width[0], self.width[1]))
        
        return img


        
        
        
def get_transforms(image_size):

    transforms_train = albumentations.Compose([
        #albumentations.Transpose(p=0.5),
        #albumentations.VerticalFlip(p=0.5),
        #albumentations.HorizontalFlip(p=0.5),
        #albumentations.RandomBrightnessContrast(p=0.5),
        albumentations.OneOf([
            albumentations.MotionBlur(),
            albumentations.MedianBlur(),
            albumentations.GaussianBlur(),
            #albumentations.GaussNoise(var_limit=(10, 30.0)),
        ], p=0.7),

        albumentations.OneOf([
            #albumentations.OpticalDistortion(distort_limit=1.0),
            #albumentations.GridDistortion(num_steps=5, distort_limit=1.),
            albumentations.ElasticTransform(alpha=3),
        ], p=0.7),

        #albumentations.CLAHE(clip_limit=4.0, p=0.7),
        #albumentations.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
        #albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=0.85),
        albumentations.Resize(image_size, image_size),
        albumentations.Cutout(max_h_size=int(image_size * 0.375), max_w_size=int(image_size * 0.375), num_holes=1, p=0.7),
        albumentations.Normalize()
    ])

    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
        albumentations.Normalize()
    ])

    return transforms_train, transforms_val



def get_meta_data(df_train, df_test):

    # One-hot encoding of anatom_site_general_challenge feature
    concat_train = df_train['anatom_site_general_challenge']
    dummies_train = pd.get_dummies(concat_train, dummy_na=True, dtype=np.uint8, prefix='site')
    concat_test = df_test['anatom_site_general_challenge']
    dummies_test = pd.get_dummies(concat_test, dummy_na=True, dtype=np.uint8, prefix='site')
    df_train = pd.concat([df_train, dummies_train.iloc[:df_train.shape[0]]], axis=1)
    df_test = pd.concat([df_test, dummies_test.iloc[:df_test.shape[0]]], axis=1)
    # Sex features
    df_train['sex'] = df_train['sex'].map({'male': 1, 'female': 0})
    df_test['sex'] = df_test['sex'].map({'male': 1, 'female': 0})
    df_train['sex'] = df_train['sex'].fillna(-1)
    df_test['sex'] = df_test['sex'].fillna(-1)
    # Age features
    df_train['age_approx'] /= 90
    df_test['age_approx'] /= 90
    df_train['age_approx'] = df_train['age_approx'].fillna(0)
    df_test['age_approx'] = df_test['age_approx'].fillna(0)
    df_train['patient_id'] = df_train['patient_id'].fillna(0)
    # n_image per user
    df_train['n_images'] = df_train.patient_id.map(df_train.groupby(['patient_id']).image_name.count())
    df_test['n_images'] = df_test.patient_id.map(df_test.groupby(['patient_id']).image_name.count())
    df_train.loc[df_train['patient_id'] == -1, 'n_images'] = 1
    df_train['n_images'] = np.log1p(df_train['n_images'].values)
    df_test['n_images'] = np.log1p(df_test['n_images'].values)
    #print("NAN present in train path ",df_train.filepath.isnull().values.any())
    #print("NAN present in test path ",df_test.filepath.isnull().values.any())
    # image size
    train_images = df_train['filepath'].values
    train_sizes = np.zeros(train_images.shape[0])
    for i, img_path in enumerate(tqdm(train_images)):
        train_sizes[i] = os.path.getsize(str(img_path))
    df_train['image_size'] = np.log(train_sizes)
    test_images = df_test['filepath'].values
    test_sizes = np.zeros(test_images.shape[0])
    for i, img_path in enumerate(tqdm(test_images)):
        test_sizes[i] = os.path.getsize(str(img_path))
    df_test['image_size'] = np.log(test_sizes)

    meta_features = ['sex', 'age_approx', 'n_images', 'image_size'] + [col for col in df_train.columns if col.startswith('site_')]
    n_meta_features = len(meta_features)

    return df_train, df_test, meta_features, n_meta_features


def get_df(kernel_type, out_dim, data_dir, data_folder, use_meta):

    # 2020 data
    print(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'train.csv'))
    df_train = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'train.csv'))
    df_train = df_train[df_train['tfrecord'] != -1].reset_index(drop=True)
    df_train['filepath'] = df_train['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/train', f'{x}.jpg'))
    print("data 20202 length df train 2",len(df_train))
    print("data 2020 length df train 2",df_train.diagnosis.value_counts())

    if 'newfold' in kernel_type:
        tfrecord2fold = {
            8:0, 5:0, 11:0,
            7:1, 0:1, 6:1,
            10:2, 12:2, 13:2,
            9:3, 1:3, 3:3,
            14:4, 2:4, 4:4,
        }
    elif 'oldfold' in kernel_type:
        tfrecord2fold = {i: i % 5 for i in range(15)}
    else:
        tfrecord2fold = {
            2:0, 4:0, 5:0,
            1:1, 10:1, 13:1,
            0:2, 9:2, 12:2,
            3:3, 8:3, 11:3,
            6:4, 7:4, 14:4,
        }
    df_train['fold'] = df_train['tfrecord'].map(tfrecord2fold)
    df_train['is_ext'] = 0

    # 2018, 2019 data (external data)
    df_train2 = pd.read_csv(os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}', 'train.csv'))
    df_train2 = df_train2[df_train2['tfrecord'] >= 0].reset_index(drop=True)
    df_train2['filepath'] = df_train2['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-isic2019-{data_folder}x{data_folder}/train', f'{x}.jpg'))
    print("data 2018 ,2019 length df train 2",len(df_train2))
    print("length df train 2",df_train2.diagnosis.value_counts())
    if 'newfold' in kernel_type:
        df_train2['tfrecord'] = df_train2['tfrecord'] % 15
        df_train2['fold'] = df_train2['tfrecord'].map(tfrecord2fold)
    else:
        df_train2['fold'] = df_train2['tfrecord'] % 5
    df_train2['is_ext'] = 1

    # Preprocess Target
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('seborrheic keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lichenoid keratosis', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('solar lentigo', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('lentigo NOS', 'BKL'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('cafe-au-lait macule', 'unknown'))
    df_train['diagnosis']  = df_train['diagnosis'].apply(lambda x: x.replace('atypical melanocytic proliferation', 'unknown'))

    if out_dim == 8:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
    elif out_dim == 4:
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('NV', 'nevus'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('MEL', 'melanoma'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('DF', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('AK', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('SCC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('VASC', 'unknown'))
        df_train2['diagnosis'] = df_train2['diagnosis'].apply(lambda x: x.replace('BCC', 'unknown'))
    else:
        raise NotImplementedError()

    # concat train data
    df_train1 = pd.concat([df_train, df_train2]).reset_index(drop=True)
    df_train1 = df_train1[df_train1.diagnosis != "unknown"].reset_index(drop=True)
    diagnosis2idx = {d: idx for idx, d in enumerate(sorted(df_train1.diagnosis.unique()))}
    print(diagnosis2idx)
    df_train1['target'] = df_train1['diagnosis'].map(diagnosis2idx)
    print("final data length df train 1",len(df_train1))
    print("final data length df train 1",df_train1.diagnosis.value_counts())    
    

    # test data
    df_train, df_test = train_test_split(df_train1, test_size=0.2,stratify=df_train1["diagnosis"], random_state = 2022)
    print("Train Length",len(df_train))
    print("Test Length",len(df_test))
    #print(df_test.info())

    #df_test = pd.read_csv(os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}', 'test.csv'))
    #df_test['filepath'] = df_test['image_name'].apply(lambda x: os.path.join(data_dir, f'jpeg-melanoma-{data_folder}x{data_folder}/test', f'{x}.jpg'))

    if use_meta:
        df_train, df_test, meta_features, n_meta_features = get_meta_data(df_train, df_test)
    else:
        meta_features = None
        n_meta_features = 0

    #print(df_train["target"])
    mel_idx = diagnosis2idx['melanoma']
    #print("mel_idx",mel_idx)
    return df_train, df_test, meta_features, n_meta_features, mel_idx
