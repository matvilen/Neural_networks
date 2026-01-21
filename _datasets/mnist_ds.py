import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image 

# Наследование от класса Dataset из torch.utils.data
class MNIST_DS(Dataset):
    def __init__(self, path, transform=None):
        self.path = path             #переданный путь до папки класса
        self.transform = transform   #трансформация - будет использоваться дальше, т.е. вместо transform будут подставляться различные функции

        self.len_dataset = 0 # длина ds (кол-во файлов в указанной папке)
        self.data_list = []  # список из кортежей (путь до файла, позиция в one-hot векторе)
        self.class_to_index = {}

        for path_dir, dir_list, file_list in os.walk(path): # см. описание ниже
            if path_dir == path:
                self.classes = sorted(dir_list) # названия папок - это и есть классы, поэтому сохраняем в таком виде
                                        # создаем словарь {class_0: 0, class_1:1 ...}
                self.class_to_index = { 
                    cls_name: i for i, cls_name, in enumerate(self.classes)
                }
                continue #создали на уровне папок классов словарь, а то что дальше это уже следующая итерация os.walk(path) с файлами
                
            cls = path_dir.split('\\')[-1]

            for name_file in file_list:
                file_path = os.path.join(path_dir, name_file)
                self.data_list.append((file_path, self.class_to_index[cls]))

            self.len_dataset += len(file_list)        
                

    def __len__(self):
        return self.len_dataset

    
    def __getitem__ (self, index):
        file_path, target = self.data_list[index]
        # sample = np.array(Image.open(file_path)) # возвращает изображение в формате PIL (ф-ция из PIL-library)

        if self.transform is not None:           # если трансформации заданы, то применим их к изображению
            sample = Image.open(file_path)
            sample = self.transform(sample)
        else:
            sample = np.array(Image.open(file_path)) # возвращает изображение в формате PIL (ф-ция из PIL-library)

        return sample, target