import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import random

class LimitedFoV(object):

    def __init__(self, fov=360.):
        self.fov = fov

    def __call__(self, x):
        angle = random.randint(0, 359)
        rotate_index = int(angle / 360. * x.shape[2])
        fov_index = int(self.fov / 360. * x.shape[2])
        if rotate_index > 0:
            img_shift = torch.zeros(x.shape)
            img_shift[:,:, :rotate_index] = x[:,:, -rotate_index:]
            img_shift[:,:, rotate_index:] = x[:,:, :(x.shape[2] - rotate_index)]
        else:
            img_shift = x

        return img_shift[:,:,:fov_index]


def input_transform_fov(size, fov):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
        LimitedFoV(fov=fov),
    ])

def input_transform(size):
    return transforms.Compose([
        transforms.Resize(size=tuple(size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225]),
    ])


# Same loader from VIGOR, modified for pytorch
class VIGOR(torch.utils.data.Dataset):
    def __init__(self, mode='', root='./VIGOR/', same_area=True, print_bool=False, args=None):
        super(VIGOR, self).__init__()

        self.args = args
        self.root = root

        self.mode = mode
        self.sat_size = [320, 320]
        self.sat_size_default = [320, 320]
        self.grd_size = [320, 640]

        if print_bool:
            print(self.sat_size, self.grd_size)

        self.sat_ori_size = [640, 640]
        self.grd_ori_size = [1024, 2048]

        if args.fov != 0:
            self.transform_query = input_transform_fov(size=self.grd_size,fov=args.fov)
        else:
            self.transform_query = input_transform(size=self.grd_size)

        self.transform_reference = input_transform(size=self.sat_size)

        self.to_tensor = transforms.ToTensor()

        self.same_area = same_area
        label_root = 'splits'

        if same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        self.train_sat_list = []
        self.train_sat_index_dict = {}
        self.delta_unit = [0.0003280724526376747, 0.00043301140280175833]
        idx = 0
        # load sat list
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.train_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', train_sat_list_fname, idx)
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        if print_bool:
            print('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            if print_bool:
                print('InputData::__init__: load', test_sat_list_fname, idx)
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        if print_bool:
            print('Test sat loaded, data size:{}'.format(self.test_sat_data_size))

        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            # load train panorama list
            train_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_train.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int32)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if not label[0] in self.train_sat_cover_dict:
                        self.train_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            if print_bool:
                print('InputData::__init__: load ', train_label_fname, idx)
        self.train_data_size = len(self.train_list)
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        if print_bool:
            print('Train grd loaded, data_size: {}'.format(self.train_data_size))

        self.__cur_test_id = 0
        self.test_list = []
        self.test_label = []
        self.test_sat_cover_dict = {}
        self.test_delta = []
        idx = 0
        for city in self.test_city_list:
            # load test panorama list
            test_label_fname = os.path.join(self.root, label_root, city, 'same_area_balanced_test.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(test_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.test_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int32)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.test_list.append(os.path.join(self.root, city, 'panorama', data[0]))
                    self.test_label.append(label)
                    self.test_delta.append(delta)
                    if not label[0] in self.test_sat_cover_dict:
                        self.test_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.test_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            if print_bool:
                print('InputData::__init__: load ', test_label_fname, idx)
        self.test_data_size = len(self.test_list)
        self.test_label = np.array(self.test_label)
        self.test_delta = np.array(self.test_delta)
        if print_bool:
            print('Test grd loaded, data size: {}'.format(self.test_data_size))

        self.train_sat_cover_list = list(self.train_sat_cover_dict.keys())


    def check_overlap(self, id_list, idx):
        output = True
        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                 if i in sat_idx:
                    output = False
                    return output
        return output

    def get_init_idx(self):
        # return random.randrange(self.train_data_size)  # sampling according to grd
        return random.choice(self.train_sat_cover_dict[random.choice(self.train_sat_cover_list)])

    def __getitem__(self, index, debug=False):
        try:
            if 'train' in self.mode:
                if 'scan' in self.mode:
                    ll = len(self.train_sat_cover_dict[self.train_sat_cover_list[index % len(self.train_sat_cover_list)]])
                    assert ll <= 2
                    idx = self.train_sat_cover_dict[self.train_sat_cover_list[index%len(self.train_sat_cover_list)]][(index//len(self.train_sat_cover_list))%ll]
                else:
                    idx = random.choice(self.train_sat_cover_dict[self.train_sat_cover_list[index%len(self.train_sat_cover_list)]])
                
                img_query = Image.open(self.train_list[idx]).convert('RGB')
                img_reference = Image.open(self.train_sat_list[self.train_label[idx][0]]).convert('RGB')
                img_query = self.transform_query(img_query)
                img_reference = self.transform_reference(img_reference)
                return img_query, img_reference, torch.tensor(idx), torch.tensor(self.train_label[idx])
            elif 'test_reference' in self.mode:
                img_path = self.test_sat_list[index]
                img_reference = Image.open(img_path).convert('RGB')
                img_reference = self.transform_reference(img_reference)
                return img_reference, torch.tensor(index)
            elif 'test_query' in self.mode:
                img_path = self.test_list[index]
                img_query = Image.open(img_path).convert('RGB')
                img_query = self.transform_query(img_query)
                return img_query, torch.tensor(index), torch.tensor(self.test_label[index])
            else:
                raise ValueError(f"Invalid mode: {self.mode}")
        except Exception as e:
            print(f"Error loading index {index} in mode {self.mode}")
            print(f"Image path: {img_path if 'img_path' in locals() else 'N/A'}")
            print(f"Error details: {str(e)}")
            raise  # 重新抛出异常以便调试

    def __len__(self):
        if 'train' in self.mode:
            return len(self.train_sat_cover_list) * 2  # one aerial image has 2 positive queries
        elif 'test_reference' in self.mode:
            return len(self.test_sat_list)
        elif 'test_query' in self.mode:
            return len(self.test_list)
        else:
            print('not implemented!')
            raise Exception


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C = np.sin(lat_A)*np.sin(lat_B) + np.cos(lat_A)*np.cos(lat_B)*np.cos(lng_A-lng_B)
    distance = R*np.arccos(C)
    return distance


# compute the distance between two locations [Lat_A,Lng_A], [Lat_B,Lng_B]
def gps2distance_matrix(Lat_A,Lng_A,Lat_B,Lng_B):
    # https://en.wikipedia.org/wiki/Great-circle_distance
    lat_A = Lat_A * np.pi/180.
    lat_B = Lat_B * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    lng_B = Lng_B * np.pi/180.
    R = 6371004.
    C1 = np.matmul(np.sin(np.expand_dims(lat_A,axis=1)), np.sin(np.expand_dims(lat_B,axis=0)))
    C2 = np.matmul(np.cos(np.expand_dims(lat_A,axis=1)),np.cos(np.expand_dims(lat_B,axis=0)))
    C2 = C2 * np.cos(np.tile(np.expand_dims(lng_A,axis=1),[1,lng_B.shape[0]])-np.tile(lng_B,[np.expand_dims(lng_A,axis=0).shape[0],1]))
    C = C1 + C2
    distance = R*np.arccos(C)
    return distance


# compute the delta unit for each reference location [Lat, Lng], 320 is half of the image width
# 0.114 is resolution in meter
# reverse equation from gps2distance: https://en.wikipedia.org/wiki/Great-circle_distance
def Lat_Lng(Lat_A, Lng_A, distance=[320*0.114, 320*0.114]):
    if distance[0] == 0 and distance[1] == 0:
        return np.zeros(2)

    lat_A = Lat_A * np.pi/180.
    lng_A = Lng_A * np.pi/180.
    R = 6371004.
    C_lat = np.cos(distance[0]/R)
    C_lng = np.cos(distance[1]/R)
    delta_lat = np.arccos(C_lat)
    delta_lng = np.arccos((C_lng-np.sin(lat_A)*np.sin(lat_A))/np.cos(lat_A)/np.cos(lat_A))
    return np.array([delta_lat * 180. / np.pi, delta_lng * 180. / np.pi])


