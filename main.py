from torch.utils.tensorboard import SummaryWriter

from predict import predict
from train_net import train_net, RDN
from settings import *

if __name__ == "__main__":
    # net = RDN.RDN(2, 3, 64, 64, 16, 8).cuda()
    # net.load_state_dict(torch.load("results/epoch_14_acc_27.0.pth"))

    # train_net(net, epochs, train_crit, valid_crit, opt, train_loader, valid_loader, scheduler)

    net.load_state_dict(torch.load("results/rdn_x2.pth"))
    predict(net, 100, sw=None)
    # sw = SummaryWriter()
    # predict(net, 0, sw)
    # sw.close()
