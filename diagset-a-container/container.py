import json
import logging
import numpy as np
import pandas as pd

from abc import ABC, abstractmethod
from pathlib import Path
from queue import Queue
from threading import Thread


IMAGENET_IMAGE_MEAN = [123.68, 116.779, 103.939]
PATCH_SIZE = 224
DEFAULT_LABEL_DICTIONARY = {'BG': 0, 'T': 1, 'N': 2, 'A': 3, 'R1': 4, 'R2': 5, 'R3': 6, 'R4': 7, 'R5': 8}


class AbstractDiagSetDataset(ABC):
    def __init__(self, root_path, partitions, magnification=40, batch_size=32, augment=True,
                 subtract_mean=True, label_dictionary=None, shuffling=True, class_ratios=None,
                 scan_subset=None, buffer_size=64):
        """
        Abstract container for DiagSet-A dataset.

        :param root_path: root directory of the dataset
        :param partitions: list containing all partitions ('train', 'validation' or 'test') that will be loaded
        :param magnification: int in [40, 20, 10, 5] describing scan magnification for which patches will be loaded
        :param batch_size: int, number of images in a single batch
        :param augment: boolean, whether to apply random image augmentations
        :param subtract_mean: boolean, whether to subtract ImageNet mean from every image
        :param label_dictionary: dict assigning int label to every text key, DEFAULT_LABEL_DICTIONARY will
               be used if it is set to None
        :param shuffling: boolean, whether to shuffle the order of batches
        :param class_ratios: dict assigning probability to each int key, specifies ratio of images from a class
               with a given key that will be loaded in each batch (note that it will not always return deterministic
               number of images per class, but will specify the probability of drawing from that class instead).
               Can be None, in which case original dataset
               ratios will be used, otherwise all dict values should sum up to one
        :param scan_subset: subset of scans that will be loaded, either list of strings with scan IDs or
               float in (0, 1), in which case a random subset of scans from given partitions will be selected
        :param buffer_size: number of images from each class that will be stored in buffer
        """
        for partition in partitions:
            assert partition in ['train', 'validation', 'test']

        self.root_path = root_path
        self.partitions = partitions
        self.magnification = magnification
        self.batch_size = batch_size
        self.augment = augment
        self.subtract_mean = subtract_mean
        self.shuffling = shuffling
        self.scan_subset = scan_subset
        self.buffer_size = buffer_size

        if label_dictionary is None:
            logging.info('Using default label dictionary...')

            self.label_dictionary = DEFAULT_LABEL_DICTIONARY
        else:
            self.label_dictionary = label_dictionary

        self.numeric_labels = list(set(self.label_dictionary.values()))

        self.buffers = {}
        self.blob_paths = {}
        self.class_distribution = {}

        for numeric_label in self.numeric_labels:
            self.buffers[numeric_label] = Queue(buffer_size)
            self.blob_paths[numeric_label] = []
            self.class_distribution[numeric_label] = 0

        self.n_images = 0

        self.blobs_path = Path(root_path) / 'blobs' / 'S' / ('%dx' % magnification)
        self.distributions_path = Path(root_path) / 'distributions' / 'S' / ('%dx' % magnification)

        assert self.blobs_path.exists()

        self.scan_names = [path.name for path in self.blobs_path.iterdir()]

        partition_scan_names = []

        for partition in self.partitions:
            partition_path = Path(root_path) / 'partitions' / 'DiagSet-A.2' / ('%s.csv' % partition)

            if partition_path.exists():
                df = pd.read_csv(partition_path)
                partition_scan_names += df['scan_id'].astype(np.str).tolist()
            else:
                raise ValueError('Partition file not found under "%s".' % partition_path)

        self.scan_names = [scan_name for scan_name in self.scan_names if scan_name in partition_scan_names]

        if self.scan_subset is not None and self.scan_subset != 1.0:
            if type(self.scan_subset) is list:
                logging.info('Using given %d out of %d scans...' % (len(self.scan_subset), len(self.scan_names)))

                self.scan_names = self.scan_subset
            else:
                if type(self.scan_subset) is float:
                    n_scans = int(self.scan_subset * len(self.scan_names))
                else:
                    n_scans = self.scan_subset

                assert n_scans > 0
                assert n_scans <= len(self.scan_names)

                logging.info('Randomly selecting %d out of %d scans...' % (n_scans, len(self.scan_names)))

                self.scan_names = list(np.random.choice(self.scan_names, n_scans, replace=False))

        logging.info('Loading blob paths...')

        for scan_name in self.scan_names:
            for string_label, numeric_label in self.label_dictionary.items():
                blob_names = map(lambda x: x.name, sorted((self.blobs_path / scan_name / string_label).iterdir()))

                for blob_name in blob_names:
                    self.blob_paths[numeric_label].append(self.blobs_path / scan_name / string_label / blob_name)

            with open(self.distributions_path / ('%s.json' % scan_name), 'r') as f:
                scan_class_distribution = json.load(f)

            self.n_images += sum(scan_class_distribution.values())

            for string_label, numeric_label in self.label_dictionary.items():
                self.class_distribution[numeric_label] += scan_class_distribution[string_label]

        if class_ratios is None:
            self.class_ratios = {}

            for numeric_label in self.numeric_labels:
                self.class_ratios[numeric_label] = self.class_distribution[numeric_label] / self.n_images
        else:
            self.class_ratios = class_ratios

        logging.info('Found %d patches.' % self.n_images)

        class_distribution_text = ', '.join(['%s: %.2f%%' % (label, count / self.n_images * 100)
                                             for label, count in self.class_distribution.items()])
        logging.info('Class distribution: %s.' % class_distribution_text)

        if self.shuffling:
            for numeric_label in self.numeric_labels:
                np.random.shuffle(self.blob_paths[numeric_label])

        for numeric_label in self.numeric_labels:
            if len(self.blob_paths[numeric_label]) > 0:
                Thread(target=self.fill_buffer, daemon=True, args=(numeric_label, )).start()

    @abstractmethod
    def batch(self):
        return

    def length(self):
        return int(np.ceil(self.n_images / self.batch_size))

    def fill_buffer(self, numeric_label):
        while True:
            for blob_path in self.blob_paths[numeric_label]:
                images = self.prepare_images(blob_path)

                for image in images:
                    self.buffers[numeric_label].put(image)

            if self.shuffling:
                np.random.shuffle(self.blob_paths[numeric_label])

    def prepare_images(self, blob_path):
        images = np.load(blob_path)

        if self.shuffling:
            np.random.shuffle(images)

        prepared_images = []

        for i in range(len(images)):
            image = images[i].astype(np.float32)

            if self.augment:
                image = self._augment(image)
            else:
                x = (image.shape[0] - PATCH_SIZE) // 2
                y = (image.shape[1] - PATCH_SIZE) // 2

                image = image[x:(x + PATCH_SIZE), y:(y + PATCH_SIZE)]

            if self.subtract_mean:
                image -= IMAGENET_IMAGE_MEAN

            prepared_images.append(image)

        prepared_images = np.array(prepared_images)

        return prepared_images

    def _augment(self, image):
        x_max = image.shape[0] - PATCH_SIZE
        y_max = image.shape[1] - PATCH_SIZE

        x = np.random.randint(x_max)
        y = np.random.randint(y_max)

        image = image[x:(x + PATCH_SIZE), y:(y + PATCH_SIZE)]

        if np.random.choice([True, False]):
            image = np.fliplr(image)

        image = np.rot90(image, k=np.random.randint(4))

        return image


class TrainingDiagSetDataset(AbstractDiagSetDataset):
    def batch(self):
        probabilities = [self.class_ratios[label] for label in self.numeric_labels]

        labels = np.random.choice(self.numeric_labels, self.batch_size, p=probabilities)
        images = np.array([self.buffers[label].get() for label in labels])

        return images, labels


class EvaluationDiagSetDataset(AbstractDiagSetDataset):
    def __init__(self, **kwargs):
        assert kwargs.get('augment', False) is False
        assert kwargs.get('shuffling', False) is False
        assert kwargs.get('class_ratios') is None

        kwargs['augment'] = False
        kwargs['shuffling'] = False
        kwargs['class_ratios'] = None

        self.current_numeric_label_index = 0
        self.current_batch_index = 0

        super().__init__(**kwargs)

    def batch(self):
        labels = []
        images = []

        for _ in range(self.batch_size):
            label = self.numeric_labels[self.current_numeric_label_index]

            while len(self.blob_paths[label]) == 0:
                self.current_numeric_label_index = (self.current_numeric_label_index + 1) % len(self.numeric_labels)

                label = self.numeric_labels[self.current_numeric_label_index]

            image = self.buffers[label].get()

            labels.append(label)
            images.append(image)

            self.current_batch_index += 1

            if self.current_batch_index >= self.class_distribution[label]:
                self.current_batch_index = 0
                self.current_numeric_label_index += 1

                if self.current_numeric_label_index >= len(self.numeric_labels):
                    self.current_numeric_label_index = 0

                    break

        labels = np.array(labels)
        images = np.array(images)

        return images, labels
