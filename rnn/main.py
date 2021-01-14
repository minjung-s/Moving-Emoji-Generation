import train_function
import torch
from utils import init_argument

if __name__ == "__main__":

    args = init_argument()
    save_path = args.save_path
    landmark_gen = train_function.trainer(args.data_path, args.t_stamp, args.batch,
    args.num_epochs, args.conditions)
    landmark_gen.eval()
    torch.save(landmark_gen, save_path)
