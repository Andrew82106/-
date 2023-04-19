import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import os
import json
from PIL import Image
from torch.utils.data.dataset import T_co
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader
from torchvision.transforms import transforms

train_data_route = './datasets/archive'

image_route = os.path.join(train_data_route, "images", "Images")
with open("./nameMap.json", "r", encoding='utf-8') as f:
    nameMap = json.load(f)


class DogPic(Dataset):
    def __init__(self, rootLoc: str, FileName: str, picSize=(224, 224)):
        super(DogPic, self).__init__()
        self.picSize = picSize
        self.label = FileName.split("-")[-1]
        self.root = os.path.join(rootLoc, "images", "Images", FileName)
        self.pic_list = os.listdir(self.root)
        self.resize = transforms.Resize(picSize)
        self.ToTensor = transforms.PILToTensor()
        self.nameMap = {"silky_terrier": 0, "Scottish_deerhound": 1, "Chesapeake_Bay_retriever": 2, "Ibizan_hound": 3, "haired_fox_terrier": 4, "Saluki": 5, "cocker_spaniel": 6, "schipperke": 7, "borzoi": 8, "Pembroke": 9, "komondor": 10, "Staffordshire_bullterrier": 11, "standard_poodle": 12, "Eskimo_dog": 13, "English_foxhound": 14, "golden_retriever": 15, "Sealyham_terrier": 16, "Japanese_spaniel": 17, "miniature_schnauzer": 18, "malamute": 19, "malinois": 20, "Pekinese": 21, "giant_schnauzer": 22, "Mexican_hairless": 23, "Doberman": 24, "standard_schnauzer": 25, "dhole": 26, "German_shepherd": 27, "Bouvier_des_Flandres": 28, "Siberian_husky": 29, "Norwich_terrier": 30, "Irish_terrier": 31, "Norfolk_terrier": 32, "Saint_Bernard": 33, "Border_terrier": 34, "briard": 35, "Tibetan_mastiff": 36, "bull_mastiff": 37, "Maltese_dog": 38, "Kerry_blue_terrier": 39, "kuvasz": 40, "Greater_Swiss_Mountain_dog": 41, "Lakeland_terrier": 42, "Blenheim_spaniel": 43, "basset": 44, "West_Highland_white_terrier": 45, "Chihuahua": 46, "Border_collie": 47, "redbone": 48, "Irish_wolfhound": 49, "bluetick": 50, "miniature_poodle": 51, "Cardigan": 52, "EntleBucher": 53, "Norwegian_elkhound": 54, "haired_pointer": 55, "Bernese_mountain_dog": 56, "papillon": 57, "Tibetan_terrier": 58, "Gordon_setter": 59, "American_Staffordshire_terrier": 60, "vizsla": 61, "kelpie": 62, "Weimaraner": 63, "miniature_pinscher": 64, "boxer": 65, "chow": 66, "Old_English_sheepdog": 67, "pug": 68, "Rhodesian_ridgeback": 69, "Scotch_terrier": 70, "Tzu": 71, "affenpinscher": 72, "whippet": 73, "Sussex_spaniel": 74, "otterhound": 75, "coated_retriever": 111, "English_setter": 77, "Italian_greyhound": 78, "Labrador_retriever": 79, "collie": 80, "cairn": 81, "Rottweiler": 82, "Australian_terrier": 83, "toy_terrier": 84, "Shetland_sheepdog": 85, "African_hunting_dog": 86, "Newfoundland": 87, "Walker_hound": 88, "Lhasa": 89, "beagle": 90, "Samoyed": 91, "Great_Dane": 92, "Airedale": 93, "bloodhound": 94, "Irish_setter": 95, "keeshond": 96, "Dandie_Dinmont": 97, "basenji": 98, "Bedlington_terrier": 99, "Appenzeller": 100, "clumber": 101, "toy_poodle": 102, "Great_Pyrenees": 103, "English_springer": 104, "Afghan_hound": 105, "Brittany_spaniel": 106, "Welsh_springer_spaniel": 107, "Boston_bull": 108, "dingo": 109, "coated_wheaten_terrier": 110, "French_bulldog": 112, "Irish_water_spaniel": 113, "Pomeranian": 114, "Brabancon_griffon": 115, "Yorkshire_terrier": 116, "groenendael": 117, "Leonberg": 118, "tan_coonhound": 119}

    def __getitem__(self, index) -> T_co:
        img_ = Image.open(os.path.join(self.root, self.pic_list[index]))
        img_ = self.resize(img_)
        img_ = self.ToTensor(img_)
        label_ = self.nameMap[self.label]
        return img_, label_

    def __len__(self):
        return len(self.pic_list)


def Generate_dataLoader():
    type_list = os.listdir(image_route)
    AllDataset = None
    print("Info: Load data")
    for file_name in type_list:
        type_name = file_name.split("-")[-1]
        print(f"Info: load dog species:{type_name}")
        dataSet_cur = DogPic(train_data_route, file_name)
        if AllDataset is not None:
            AllDataset += dataSet_cur
        else:
            AllDataset = dataSet_cur

    dataset_size = len(AllDataset)
    print(f"Info: success load all picture (len={dataset_size})")
    test_ratio = 0.3
    train_ratio = 1 - test_ratio
    train_size = int(train_ratio * dataset_size)

    # 创建训练集和测试集的索引
    indices = list(range(dataset_size))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    # 创建SubsetRandomSampler
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    # 创建DataLoader
    train_loader = DataLoader(AllDataset, batch_size=32, sampler=train_sampler, num_workers=0)
    test_loader = DataLoader(AllDataset, batch_size=32, sampler=test_sampler, num_workers=0)

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = Generate_dataLoader()
    for img, label in train_loader:
        print(label)
