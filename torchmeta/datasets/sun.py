"""
SUN original website (where to download images and attributes): http://cs.brown.edu/~gmpatter/sunattributes.html
split rule：https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
reference: "Zero-Shot Learning - A Comprehensive Evaluation of the Good, the Bad and the Ugly "
"""

import os
import pickle
import scipy.io
from PIL import Image
import h5py
import json
import torch

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_url, download_file_from_google_drive


class SUN(CombinationMetaDataset):
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
        dataset = SUNClassDataset(root, meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            transform=transform, class_augmentations=class_augmentations,
            download=download)
        super(SUN, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class SUNClassDataset(ClassDataset):
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
        super(SUNClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)
       
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform

        self.split_labels_filename = os.path.join(self.root, self.labels_filename.format(self.meta_split))
        self.image_id_name_filename = os.path.join(self.root, self.attribute_dir, self.image_id_name_filename)
        self.image_attribute_labels_filename = os.path.join(self.root, self.attribute_dir, self.image_attribute_labels_filename)

        self._data = None
        self._labels = None

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError()
        self._num_classes = len(self.labels)

    def __getitem__(self, index):
        class_name = self.labels[index % self.num_classes]
        data = self.data[class_name]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)

        return SUNDataset(index, data, class_name, transform=transform, target_transform=target_transform)

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
            f.extractall(self.root)

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
            f.extractall(self.root)

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


class SUNDataset(Dataset):
    def __init__(self, index, data, class_name, transform=None, target_transform=None):
        super(SUNDataset, self).__init__(index, transform=transform,
            target_transform=target_transform)

        self.data = data    # dict
        self.class_name = class_name
        self.image_names = list(data.keys())
    
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):

        image_name = self.image_names[index]
        image = self.data[image_name]
        target = self.class_name

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return (image, target)