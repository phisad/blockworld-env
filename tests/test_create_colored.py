import unittest

from torchexpresso.configs import ExperimentConfigLoader
from torchexpresso.runners import Processor

DATASET_DIRECTORY = "C:/Development/data/blockworld-random-2d"


class Test(unittest.TestCase):
    config_directory_path = "../configs"

    def test_create_multi_random_block2d_train(self):
        Test.processing("create-multi-random-block2d-colored",
                        dataset_directory=DATASET_DIRECTORY,
                        split_name="train",
                        num_samples_per_epoch="1000")

    def test_create_multi_random_block2d_dev(self):
        Test.processing("create-multi-random-block2d-colored",
                        dataset_directory=DATASET_DIRECTORY,
                        split_name="dev",
                        num_samples_per_epoch="100")

    def test_create_single_random_block2d_dev(self):
        Test.processing("create-single-random-block2d-colored",
                        dataset_directory=DATASET_DIRECTORY,
                        split_name="dev",
                        num_samples_per_epoch="100")

    def test_create_single_random_block2d_train(self):
        Test.processing("create-single-random-block2d-colored",
                        dataset_directory=DATASET_DIRECTORY,
                        split_name="train",
                        num_samples_per_epoch="1000")

    def test_create_single_random_block2d_patches_dev(self):
        Test.processing("create-single-random-block2d-colored-patches",
                        dataset_directory=DATASET_DIRECTORY,
                        split_name="dev",
                        num_samples_per_epoch="100")

    def test_create_single_random_block2d_patches_train(self):
        Test.processing("create-single-random-block2d-colored-patches",
                        dataset_directory=DATASET_DIRECTORY,
                        split_name="train",
                        num_samples_per_epoch="1000")

    @staticmethod
    def processing(experiment_name, dataset_directory, split_name, num_samples_per_epoch):
        experiment_config = ExperimentConfigLoader(Test.config_directory_path) \
            .with_placeholders(dataset_directory=dataset_directory,
                               num_samples_per_epoch=num_samples_per_epoch,
                               split_name=split_name) \
            .load(experiment_name)
        processor = Processor.from_config(experiment_config, split_name)
        processor.perform()


if __name__ == '__main__':
    unittest.main()
