import os
import sys
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from src.utilities import get_config
from src.mocnn.MOCNN import MO_CNN
from src.mocnn.dataloader import LoadData
from src.mocnn.helpers import mae, mase, un_normalize, count_parameters, dataset_to_np


def main(argv=sys.argv[1:]):
    config = get_config(argv)

    save_dir = config["save_dir"]
    work_dir = config["work_dir"]
    model_name = config["model_name"]
    test_set = config["test_set"]
    train = config["train"]
    test = config["test"]

    print("Config Loaded")

    seed = None
    np.random.seed(seed)

    if train:
        model = MO_CNN()
    else:
        model = MO_CNN.load_from_checkpoint(
            checkpoint_path=f"{work_dir}models/{model_name}.ckpt"
        )

    model = model.float()

    if config["gpus"]:
        trainer = pl.Trainer(max_epochs=480, gpus=-1, progress_bar_refresh_rate=0)
    else:
        trainer = pl.Trainer(max_epochs=480, progress_bar_refresh_rate=0)

    if train:
        network_size = count_parameters(model)
        print(f"There are {network_size} tunable parameters in this model")

        train_data = LoadData(dir, "new_train")
        valid_data = LoadData(dir, "valid")

        train_loader = DataLoader(
            train_data, batch_size=32, pin_memory=True, num_workers=4
        )
        valid_loader = DataLoader(
            valid_data, batch_size=32, pin_memory=True, num_workers=4
        )

        print("data loaded")

        if valid_loader:
            trainer.fit(
                model, train_dataloaders=train_loader, val_dataloaders=valid_loader
            )
        else:
            trainer.fit(model, train_dataloaders=train_loader)

        trainer.save_checkpoint(f"models/{model_name}.ckpt")

        del train_data
        del valid_data
        del train_loader
        del valid_loader

        print("training complete!")
        print("loading test data!")

        test_data = LoadData(dir, "test")
        test_loader = DataLoader(
            test_data, batch_size=32, pin_memory=True, num_workers=4
        )

        trainer.test(model, dataloaders=test_loader)

        predictions = model.return_results().cpu().numpy()

        np.save(
            os.path.join(save_dir, "output", f"{model_name}_predictions.npy"),
            predictions,
        )

        x, y_test = dataset_to_np(test_data)

        MASE = mase(predictions, y_test)
        MAE = mae(predictions, y_test)

        print(f"MASE is {MASE * 100} and MAE is {MAE * 100}")
    elif test:
        print("expanding boundary")
        # test_data = LoadData(dir, 'boundary')
        # test_loader = DataLoader(test_data, batch_size=32, pin_memory=True, num_workers=4)
        # fp = os.path.join(cwd, 'inputs', '51_51_51/truncated_51_stats.h5')
        # dat = h5py.File(os.path.join(dir, 'test_stats_u.h5'))
        # x = np.array(dat['2PS'])

        fp = os.path.join(work_dir, "data", "inputs", "boundary_interpolate_stats.npy")
        x = np.load(fp)
        x = x[:, None, ...].real

        x = torch.as_tensor(x).float()

        model.eval()
        predictions = model(x).detach().numpy()
        predictions = un_normalize(
            predictions, np.array(((8067.9, 2307), (161.5, 46.15)))
        )
        print(predictions)

        np.save(
            os.path.join(
                save_dir, "protocol", f"{model_name}_boundary_cnn_updating.npy"
            ),
            predictions,
        )

    return model


if __name__ == "__main__":
    main()
