from torch.optim import SGD, Adam


def get_optimizer(parameters, opt):
    if opt.optimizer == "SGD":
        return SGD(
            params=parameters,
            lr=opt.learning_rate,
            momentum=opt.momentum,
            weight_decay=opt.weight_decay
        )
    elif opt.optimizer == "Adam":
        return Adam(
            params=parameters,
            lr=opt.learning_rate,
            weight_decay=opt.weight_decay,
        )
    else:
        raise ValueError(f"Specified optimizer name '{opt.optimizer}' is not valid.")
