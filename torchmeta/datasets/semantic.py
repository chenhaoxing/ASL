'''
CUB original website: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
auxiliary modality data can be obtained from the repository of Learning Deep Representations of Fine-grained Visual Descriptions：https://github.com/reedscot/cvpr2016

SUN original website (where to download images and attributes): http://cs.brown.edu/~gmpatter/sunattributes.html
split rule：https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
reference: "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly"
'''

import numpy as np
from PIL import Image
import os
import io
import json
import glob
import h5py
import torch
import pickle
import scipy.io

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_url, download_file_from_google_drive
from torchmeta.datasets.utils import get_asset


class CUBMM(CombinationMetaDataset):
    """
    The Caltech-UCSD Birds dataset, introduced in [1]. This dataset is based on
    images from 200 species of birds from the Caltech-UCSD Birds dataset [2].

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `cub` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way" 
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`, 
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `torchmeta.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g.
        `torchmeta.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root 
        directory (under the `cub` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [2]. The dataset contains images from 200
    classes. The meta train/validation/test splits are over 100/50/50 classes.
    The splits are taken from [3] ([code](https://github.com/wyharveychen/CloserLookFewShot)
    for reproducibility).

    References
    ----------
    .. [1] Hilliard, N., Phillips, L., Howland, S., Yankov, A., Corley, C. D.,
           Hodas, N. O. (2018). Few-Shot Learning with Metric-Agnostic Conditional
           Embeddings. (https://arxiv.org/abs/1802.04376)
    .. [2] Wah, C., Branson, S., Welinder, P., Perona, P., Belongie, S. (2011).
           The Caltech-UCSD Birds-200-2011 Dataset
           (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
    .. [3] Chen, W., Liu, Y. and Kira, Z. and Wang, Y. and  Huang, J. (2019).
           A Closer Look at Few-shot Classification. International Conference on
           Learning Representations (https://openreview.net/forum?id=HkxLXnAcFQ)

    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = CUBClassDataset(root, meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download)
        super(CUBMM, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class CUBClassDataset(ClassDataset):
    folder = 'cub'

    # Google Drive ID from http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
    gdrive_id = '1hbzc_P1FuxMkcabkgn9ZKinBwW683j45'
    tgz_filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'
    image_folder = 'CUB_200_2011/images'

    filename = '{0}_data.hdf5'
    filename_labels = '{0}_labels.json'

    assets_dir = 'assets'
    text_dir = 'text_c10'
    attribute_dir = 'attributes'
    class_attribute_filename_labels = 'class_attribute_labels_continuous.txt'
    image_id_name_filename = 'images.txt'
    image_attribute_filename_labels = 'image_attribute_labels.txt'
    classes_filename = 'classes.txt'
    attributes_dim = 312

    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False):
        super(CUBClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)

        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_filename = os.path.join(self.root,
            self.filename.format(self.meta_split))
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self.class_attribute_labels_filename = os.path.join(root, self.assets_dir, self.folder, self.attribute_dir, self.class_attribute_filename_labels)
        self.image_id_name_filename = os.path.join(root, self.assets_dir, self.folder, self.attribute_dir, self.image_id_name_filename)
        self.image_attribute_labels_filename = os.path.join(root, self.assets_dir, self.folder, self.attribute_dir, self.image_attribute_filename_labels)
        self.classes_filename = os.path.join(root, self.assets_dir, self.folder, self.classes_filename)

        self._data_file = None
        self._data = None
        self._labels = None

        self.classes_attributes_dict = self.get_classes_attributes_dict()

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('CUB integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        class_attributes = self.classes_attributes_dict[label]
        data = self.data[label]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return CUBDataset(index, data, label, class_attributes,
            transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        if self._data is None:
            self._data_file = h5py.File(self.split_filename, 'r')
            self._data = self._data_file['datasets']
        return self._data
    
    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels

    def get_images_attributes_dict(self):
        # get a dict that contains attributes of all images: {class_name: {image_name: attribute}}
        image_id_name_dict = {}
        with open(self.image_id_name_filename, 'r', encoding='utf-8') as f:
            for line in f:
                content = line.strip('\n').split(' ')
                image_id_name_dict[content[0]] = content[1].split('/')

        image_attribute_table = []
        with open(self.image_attribute_labels_filename, 'r', encoding='utf-8') as f:
            for line in f:
                image_attribute_table.append(line.strip('\n').split(' '))

        images_attributes_dict = {}
        for i in range(len(image_attribute_table) // self.attributes_dim):
            class_name = image_id_name_dict[str(i+1)][0]
            if class_name not in images_attributes_dict.keys():
                images_attributes_dict[class_name] = {}
            image_name = image_id_name_dict[str(i+1)][1].replace('.jpg', '')
            if image_name not in images_attributes_dict[class_name].keys():
                images_attributes_dict[class_name][image_name] = []
                for j in range(self.attributes_dim):
                    images_attributes_dict[class_name][image_name].append(image_attribute_table[self.attributes_dim * i + j][2])
                images_attributes_dict[class_name][image_name] = np.array([float(item) for item in images_attributes_dict[class_name][image_name]])    # str to int, list to numpy
        return images_attributes_dict

    def get_classes_attributes_dict(self):
        # get a dict that contains attributes of all classes, {class_name: [attribute_value_1, ...]}
        class_attributes_dict = {}
        class_attributes = []
        with open(self.class_attribute_labels_filename, 'r', encoding='utf-8') as f:
            for line in f:
                single_class_attribute = line.strip('\n').split(' ')
                single_class_attribute = np.array([float(item)/100 for item in single_class_attribute])    # [0, 100] -> [0, 1]
                class_attributes.append(single_class_attribute)
        with open(self.classes_filename, 'r', encoding='utf-8') as f:
            for line in f:
                class_id_name = line.strip('\n').split(' ')
                class_attributes_dict[class_id_name[1]] = class_attributes[int(class_id_name[0])-1]
        del class_attributes
        return class_attributes_dict

    def _check_integrity(self):
        return (os.path.isfile(self.split_filename)
            and os.path.isfile(self.split_filename_labels)
            and os.path.isfile(self.class_attribute_labels_filename)
            and os.path.isfile(self.image_id_name_filename))

    def close(self):
        if self._data_file is not None:
            self._data_file.close()
            self._data_file = None
            self._data = None

    def download(self):
        import tarfile
        import shutil
        import glob
        from tqdm import tqdm

        if self._check_integrity():
            return

        tgz_filename = os.path.join(self.root, self.tgz_filename)
        if not os.path.isfile(tgz_filename):
            download_file_from_google_drive(self.gdrive_id, self.root,
                self.tgz_filename, md5=self.tgz_md5)

        tgz_filename = os.path.join(self.root, self.tgz_filename)
        with tarfile.open(tgz_filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)
        image_folder = os.path.join(self.root, self.image_folder)

        for split in ['train', 'val', 'test']:
            filename = os.path.join(self.root, self.filename.format(split))
            if os.path.isfile(filename):
                continue

            labels = get_asset(self.folder, '{0}.json'.format(split))
            labels_filename = os.path.join(self.root, self.filename_labels.format(split))
            with open(labels_filename, 'w') as f:
                json.dump(labels, f)

            with h5py.File(filename, 'w') as f:
                group = f.create_group('datasets')
                dtype = h5py.special_dtype(vlen=np.uint8)
                for i, label in enumerate(tqdm(labels, desc=filename)):
                    images = glob.glob(os.path.join(image_folder, label, '*.jpg'))
                    images.sort()
                    dataset = group.create_dataset(label, (len(images),), dtype=dtype)
                    for i, image in enumerate(images):
                        with open(image, 'rb') as f:
                            array = bytearray(f.read())
                            dataset[i] = np.asarray(array, dtype=np.uint8)

        tar_folder, _ = os.path.splitext(tgz_filename)
        if os.path.isdir(tar_folder):
            shutil.rmtree(tar_folder)

        attributes_filename = os.path.join(self.root, 'attributes.txt')
        if os.path.isfile(attributes_filename):
            os.remove(attributes_filename)


class CUBDataset(Dataset):
    def __init__(self, index, data, label, class_attributes,
                 transform=None, target_transform=None):
        super(CUBDataset, self).__init__(index, transform=transform,
                                         target_transform=target_transform)
        self.data = data
        self.label = label
        self.class_attributes = class_attributes

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image = Image.open(io.BytesIO(self.data[index])).convert('RGB')
        target = self.label

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return {
            'images': image,
            'class_attributes': self.class_attributes,
            'targets': target
        }


class SUNMM(CombinationMetaDataset):
    """
    The SUN Attribute dataset (SUN), introduced in [1]. This dataset contains 14,340 images from 717 categories. 102 real-valued labels ranging from 0-1 for each image is provided. The meta train/validation/test splits are taken from [2] for reproducibility.

    Parameters
    ----------
    root : string
        Root directory where the dataset folder `sun` exists.

    num_classes_per_task : int
        Number of classes per tasks. This corresponds to "N" in "N-way" 
        classification.

    meta_train : bool (default: `False`)
        Use the meta-train split of the dataset. If set to `True`, then the
        arguments `meta_val` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_val : bool (default: `False`)
        Use the meta-validation split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_test` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_test : bool (default: `False`)
        Use the meta-test split of the dataset. If set to `True`, then the 
        arguments `meta_train` and `meta_val` must be set to `False`. Exactly one 
        of these three arguments must be set to `True`.

    meta_split : string in {'train', 'val', 'test'}, optional
        Name of the split to use. This overrides the arguments `meta_train`, 
        `meta_val` and `meta_test` if all three are set to `False`.

    transform : callable, optional
        A function/transform that takes a `PIL` image, and returns a transformed 
        version. See also `torchvision.transforms`.

    target_transform : callable, optional
        A function/transform that takes a target, and returns a transformed 
        version. See also `torchvision.transforms`.

    dataset_transform : callable, optional
        A function/transform that takes a dataset (ie. a task), and returns a 
        transformed version of it. E.g. `mmfsl.transforms.ClassSplitter()`.

    class_augmentations : list of callable, optional
        A list of functions that augment the dataset with new classes. These classes 
        are transformations of existing classes. E.g.
        `mmfsl.transforms.HorizontalFlip()`.

    download : bool (default: `False`)
        If `True`, downloads the pickle files and processes the dataset in the root 
        directory (under the `sun` folder). If the dataset is already 
        available, this does not download/process the dataset again.

    Notes
    -----
    The dataset is downloaded from [here](http://cs.brown.edu/~gmpatter/sunattributes.html). 
    The meta train/validation/test splits are over 580/65/72 classes.

    References
    ----------
    ..  [1] Genevieve Patterson, Chen Xu, Hang Su, James Hays. The SUN Attribute Database: 
           Beyond Categories for Deeper Scene Understanding. IJCV 2014. 

    ..  [2] Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly.
    """
    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False):
        dataset = SUNClassMMDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(SUNMM, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class SUNClassMMDataset(ClassDataset):
    folder = 'sun'
    images_url = 'http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB_Images.tar.gz'
    attributes_url = 'http://cs.brown.edu/~gmpatter/Attributes/SUNAttributeDB.tar.gz'
    labels_filename = '{0}classes.txt'

    assets_dir = 'assets'
    image_dir = 'ordered_images'
    attribute_dir = 'attributes'
    image_id_name_filename = 'images.mat'
    image_attribute_labels_filename = 'attributeLabels_continuous.mat'
    attributes_dim = 102


    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                meta_split=None, transform=None, class_augmentations=None,
                download=False):
        super(SUNClassMMDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
       
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_labels_filename = os.path.join(root, self.assets_dir, self.folder, self.labels_filename.format(self.meta_split))
        self.image_id_name_filename = os.path.join(self.root, self.attribute_dir, self.image_id_name_filename)
        self.image_attribute_labels_filename = os.path.join(self.root, self.attribute_dir, self.image_attribute_labels_filename)

        self._data = None
        self._labels = None

        if download:
            self.download()

        # get a dict as {class_name: {image_name: attribute}}
        self.images_attributes_dict = self.get_images_attributes_dict()

        if not self._check_integrity():
            raise RuntimeError('SUN integrity check failed')
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        image_attributes = self.images_attributes_dict[class_name]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return SUNMMDataset(index, data, class_name, image_attributes, transform=transform, target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def data(self):
        '''
        from attributes file
        get a dict as {'class_name': {'image_name': Image}}
        '''
        if self._data is None:
            self._data = {}
            image_id_name_file = scipy.io.loadmat(self.image_id_name_filename)['images']
            for i_image in range(len(image_id_name_file)):
                name_list = image_id_name_file[i_image][0][0].split('/')
                if len(name_list) == 4:
                    class_name = '{0}_{1}'.format(name_list[1], name_list[2])
                else:    # len(name_list) == 3
                    class_name = name_list[1]
                filename = name_list[-1]
                if class_name not in self._data.keys():
                    self._data[class_name] = {}
                class_images_dir = os.path.join(self.root, self.image_dir, class_name)
                file_path = os.path.join(class_images_dir, filename)
                image_name = filename.replace('.jpg', '')
                image = Image.open(file_path).convert('RGB')
                self._data[class_name][image_name] = image.copy()
                image.close()

        return self._data


    @property
    def labels(self):
        '''
        get all class names of train/valid/test
        read the .txt file and return a list
        '''
        if self._labels is None:
            self._labels = []
            with open(self.split_labels_filename, 'r') as f:
                for line in f:
                    self._labels.append(line.strip('\n'))
        return self._labels


    def get_images_attributes_dict(self):
        # {class_name: {image_name: attribute}}
        images_attributes_dict = {}
        image_id_name_file = scipy.io.loadmat(self.image_id_name_filename)['images']
        image_attribute_labels_file = scipy.io.loadmat(self.image_attribute_labels_filename)['labels_cv']
        for i_image in range(len(image_id_name_file)):
            name_list = image_id_name_file[i_image][0][0].split('/')
            if len(name_list) == 4:
                class_name = '{0}_{1}'.format(name_list[1], name_list[2])
            else:    # len(name_list) == 3
                class_name = name_list[1]
            image_name = name_list[-1]
            if class_name not in images_attributes_dict.keys():
                images_attributes_dict[class_name] = {}
            image_name = image_name.replace('.jpg', '')
            if image_name not in images_attributes_dict[class_name].keys():
                images_attributes_dict[class_name][image_name] = image_attribute_labels_file[i_image]
        return images_attributes_dict


    def _check_integrity(self):
        return (os.path.isfile(self.image_id_name_filename)
            and os.path.isfile(self.image_attribute_labels_filename))


    def download(self):
        import tarfile
        import shutil

        if self._check_integrity():
            return

        # download attributes
        attributes_filename = os.path.basename(self.attributes_url)
        download_url(self.attributes_url, self.root, filename=attributes_filename)

        attributes_tgz_filename = os.path.join(self.root, attributes_filename)
        with tarfile.open(attributes_tgz_filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)

        if os.path.isfile(attributes_tgz_filename):
            os.remove(attributes_tgz_filename)

        attributes_original_dir = os.path.join(self.root, attributes_filename.split('.')[0])
        attributes_final_dir = os.path.join(self.root, self.attribute_dir)
        os.rename(attributes_original_dir, attributes_final_dir)

        # download images
        images_filename = os.path.basename(self.images_url)
        download_url(self.images_url, self.root, filename=images_filename)

        images_tgz_filename = os.path.join(self.root, images_filename)
        with tarfile.open(images_tgz_filename, 'r') as f:
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, self.root)

        if os.path.isfile(images_tgz_filename):
            os.remove(images_tgz_filename)

        images_original_dir = os.path.join(self.root, 'images')
        images_final_dir = os.path.join(self.root, self.image_dir)

        for dir_name in os.listdir(images_original_dir):
            if dir_name in ['misc', 'outliers']:
                continue
            cur_dir = os.path.join(images_original_dir, dir_name)
            for child_dir_name in os.listdir(cur_dir):
                cur_child_dir = os.path.join(cur_dir, child_dir_name) 
                first_child_name = os.listdir(cur_child_dir)[0]
                if os.path.isdir(os.path.join(cur_child_dir, first_child_name)):    # cur_dir contains dir
                    for child_name in os.listdir(cur_child_dir):
                        source_dir = os.path.join(cur_child_dir, child_name) 
                        target_dir = target_dir = os.path.join(images_final_dir, '{}_{}'.format(child_dir_name, child_name))
                        if not os.path.exists(target_dir):
                            shutil.copytree(source_dir, target_dir)
                else:    # cur_child_dir only contains images
                    target_dir = os.path.join(images_final_dir, child_dir_name)
                    if not os.path.exists(target_dir):
                        shutil.copytree(cur_child_dir, target_dir)

        if os.path.exists(images_original_dir):
            shutil.rmtree(images_original_dir)

        # delete dirs that have only 1 image
        for dir_name in ['barbershop', 'distillery', 'ice_cream_parlor', 'police_station', 
            'roller_skating_rink_indoor', 'volleyball_court_indoor']:
            delete_dir = os.path.join(images_final_dir, dir_name)
            if os.path.exists(delete_dir):
                shutil.rmtree(delete_dir)


class SUNMMDataset(Dataset):
    def __init__(self, index, data, class_name, image_attributes, transform=None, target_transform=None):
        super(SUNMMDataset, self).__init__(index, transform=transform,
            target_transform=target_transform)

        self.data = data    # dict
        self.class_name = class_name
        self.image_attributes = image_attributes    # dict
        self.image_names = list(data.keys())
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        image = self.data[image_name]
        image_attribute = self.image_attributes[image_name]
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # to Tensor
        image_attribute = torch.Tensor(image_attribute)

        return {
            'images': image,
            'image_attributes': image_attribute,
            'targets': target
        }