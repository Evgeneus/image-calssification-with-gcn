import torch


def save_model(args, model):
    # To save a DataParallel model generically, save the model.module.state_dict().
    # This way, you have the flexibility to load the model any way you want to any device you want.
    out = args.out_dir + "checkpoint_{}.tar".format(args.current_epoch)
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out)
    else:
        torch.save(model.state_dict(), out)


def load_model(args, model):
    model_path = "../logs/{}/pretrained/checkpoint_{}.tar".format(args.experiment_id, args.checkpoin_to_test)
    model.load_state_dict(torch.load(model_path))

    return model
