import argparse

parser = argparse.ArgumentParser(description="Train a Super Resolution GAN.")
parser.add_argument("--tr_path", type=str, default="DIV2K_train_HR",
                    help="Path to folder containing folder of training images.")
parser.add_argument("--val_path", type=str, default="DIV2K_valid_HR",
                    help="Path to folder containing folder of validation images.")
parser.add_argument("--epochs", type=int, default=1,
                    help="Number of epochs for training.")
parser.add_argument("--generator_lr", type=float, default=1e-3,
                    help="Generator learning rate.")
parser.add_argument("--discriminator_lr", type=float, default=1e-4,
                    help="Discriminator learning rate.")
parser.add_argument("--train_batch_size", type=int, default=8,
                    help="Batch size of training data loader.")
parser.add_argument("--validation_batch_size", type=int, default=8,
                    help="Batch size of validation data loader.")
parser.add_argument("--train_crop", type=int, default=64,
                    help="Image crop size for training.")
parser.add_argument("--validation_crop", type=int, default=64,
                    help="Image crop size for validation.")
parser.add_argument("--resume_path", type=str, default=None,
                    help="Path to a checkpoint from which you want to resume training.")
parser.add_argument("--discriminator_num_features", type=int, default=64,
                    help="Number of features of discriminator network.")
parser.add_argument("--num_workers", type=int, default=8,
                    help="Number of workers for data loaders.")
parser.add_argument("--l1_coeff", type=float, default=1.0,
                    help="L1 loss coefficient for supervised criterion.")
parser.add_argument("--vgg_coeff", type=float, default=1.0,
                    help="VGG features loss coefficient for supervised criterion.")
parser.add_argument("--gan_coeff", type=float, default=0.1,
                    help="Generator criterion coefficient for total generator loss.")
parser.add_argument("--max_images_log", type=int, default=10,
                    help="Maximum amount of validation images logged to tensorboard.")
parser.add_argument("--expand_on", type=str, default=None,
                    help="Use previously finished model and further train with it.")
parser.add_argument("--every_n", type=int, default=10,
                    help="Periodicity with which mini batch training stats are to be logged.")
parser.add_argument("--json", type=str, default=None,
                    help="JSON file with same possible arguments as parser.")
parser.add_argument("--scheduler", type=float, default=None, nargs="+",
                    help="Scheduler for training session.")
parser.add_argument("--warmup", type=float, default=None, nargs="+",
                    help="Warmup scheduler for training session.")
parser.add_argument("--best_metric", type=float, default=-1,
                    help="Minimum metric value required to save model.")
