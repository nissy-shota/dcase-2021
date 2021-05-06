from itertools import chain

import torch
import numpy

import util

CONFIG = util.load_yaml("./config.yaml")

def concat_features(file_list):
    """
    Extract features from audio files and then concate them into an array.
    """
    # calculate the number of dimensions
    n_features = []
    for file_id, file_name in enumerate(file_list):

        # extract feature from audio file.
        feature = util.extract_feature(file_name, CONFIG["feature"])
        feature = feature[:: CONFIG["feature"]["n_hop_frames"], :]

        if file_id == 0:
            features = numpy.zeros(
                (
                    len(file_list) * feature.shape[0],
                    CONFIG["feature"]["n_mels"] * CONFIG["feature"]["n_frames"],
                ),
                dtype=float,
            )

        features[
            feature.shape[0] * file_id : feature.shape[0] * (file_id + 1), :
        ] = feature

        n_features.append(feature.shape[0])

    return features, n_features

def flatten(nested_list):
    """
    Flatten list.
    """
    return list(chain.from_iterable(nested_list))

class DcaseDataset(torch.utils.data.Dataset):
    """
    Prepare dataset.
    """

    def __init__(self, unique_section_names, target_dir, mode):
        super().__init__()

        n_files_ea_section = []  # number of files for each section
        n_vectors_ea_file = []  # number of vectors for each file
        data = numpy.empty(
            (0, CONFIG["feature"]["n_frames"] * CONFIG["feature"]["n_mels"]),
            dtype=float,
        )

        for section_name in unique_section_names:
            # get file list for each section
            # all values of y_true are zero in training
            print("target_dir : %s" % (target_dir + "_" + section_name))
            files, _ = util.file_list_generator(
                target_dir=target_dir,
                section_name=section_name,
                dir_name="train",
                mode=mode,
            )
            print("number of files : %s" % (str(len(files))))

            n_files_ea_section.append(len(files))

            # extract features from audio files and
            # concatenate them into Numpy array.
            features, n_features = concat_features(files)

            data = numpy.append(data, features, axis=0)
            n_vectors_ea_file.append(n_features)

        n_vectors_ea_file = flatten(n_vectors_ea_file)

        # make target labels for conditioning
        # they are not one-hot vector!
        labels = numpy.zeros((data.shape[0]), dtype=int)
        start_index = 0
        for section_index in range(unique_section_names.shape[0]):
            for file_id in range(n_files_ea_section[section_index]):
                labels[
                    start_index : start_index + n_vectors_ea_file[file_id]
                ] = section_index
                start_index += n_vectors_ea_file[file_id]

        # 1D vector to 2D image (1ch)
        self.data = data.reshape(
            (
                data.shape[0],
                1,  # number of channels
                CONFIG["feature"]["n_frames"],
                CONFIG["feature"]["n_mels"],
            )
        )

        self.labels = labels

    def __len__(self):
        return self.data.shape[0]  # return num of samples

    def __getitem__(self, index):
        sample = self.data[index, :]
        label = self.labels[index]

        return sample, label