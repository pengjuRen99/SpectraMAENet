
from torch.utils.tensorboard import SummaryWriter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
import torch
from torch.utils.data.dataloader import default_collate
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import click
from typing import Optional, Callable, Union, Tuple
from scipy.optimize import linear_sum_assignment
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from utils.data import Pathogenic_finetune, Pathogenic_dataloader, train_transform, valid_transform
from sklearn.model_selection import train_test_split
from model.sdae import StackedDenoisingAutoEncoder
from model.dec import DEC
import model.model as ae

    

@click.command()
@click.option(
    "--cuda", help="whether to use CUDA (default False).", type=bool, default=True
)
@click.option(
    "--batch-size", help="training batch size (default 256).", type=int, default=256
)
@click.option(
    "--pretrain-epochs",
    help="number of pretraining epochs (default 300).",
    type=int,
    default=100,
)
@click.option(
    "--finetune-epochs",
    help="number of finetune epochs (default 500).",
    type=int,
    default=100,
)


def main(cuda, batch_size, pretrain_epochs, finetune_epochs,):
    writer = SummaryWriter()  # create the TensorBoard object
    # callback function to call during training, uses writer from the scope

    def training_callback(epoch, lr, loss, validation_loss):
        writer.add_scalars(
            "data/autoencoder",
            {"lr": lr, "loss": loss, "validation_loss": validation_loss,},
            epoch,
        )

    spectra_X = np.load('D:/codes/Raman_code/Dataset/bacteria_ID/'+'X_reference.npy')
    spectra_y = np.load('D:/codes/Raman_code/Dataset/bacteria_ID/'+'y_reference.npy')
    train_X, valid_X, train_y, valid_y = train_test_split(spectra_X, spectra_y, test_size=0.2, random_state=42)
    train_set = Pathogenic_finetune(train_X, train_y, transform=train_transform)
    valid_set = Pathogenic_finetune(valid_X, valid_y, transform=valid_transform)

    autoencoder = StackedDenoisingAutoEncoder(
        [1000, 500, 500, 2000, 30], final_activation=None
    )

    if cuda:
        autoencoder.cuda()
    print('Pretraining stage.')
    ae.pretrain(
        train_set,
        autoencoder,
        cuda=cuda,
        validation=valid_set,
        epochs=pretrain_epochs,
        batch_size=batch_size,
        optimizer=lambda model: SGD(model.parameters(), lr=0.1, momentum=0.9),
        scheduler=lambda x: StepLR(x, 100, gamma=0.1),
        corruption=0.2,
    )
    print("DEC stage.")
    model = DEC(cluster_number=30, hidden_dimension=30, encoder=autoencoder.encoder)
    if cuda:
        model.cuda()
    dec_optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    train(
            dataset=train_set,
            model=model,
            epochs=finetune_epochs,
            batch_size=batch_size,
            optimizer=dec_optimizer,
            stopping_delta=0.000001,
            cuda=cuda,
        )
    predicted, actual = predict(
        train_set, model, 1024, silent=True, return_actual=True, cuda=cuda
    )


    actual = actual.cpu().numpy()
    predicted = predicted.cpu().numpy()
    reassignment, accuracy = cluster_accuracy(actual, predicted)
    print("Final DEC accuracy: %s" % accuracy)

def train(
    dataset,
    model: torch.nn.Module,
    epochs: int,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    stopping_delta: Optional[float] = None,
    collate_fn=default_collate,
    cuda: bool = True,
    sampler: Optional[torch.utils.data.sampler.Sampler] = None,
    silent: bool = False,
    update_freq: int = 10,
    evaluate_batch_size: int = 1024,
    update_callback: Optional[Callable[[float, float], None]] = None,
    epoch_callback: Optional[Callable[[int, torch.nn.Module], None]] = None,
) -> None:
     
    static_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=sampler,
        shuffle=False,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        sampler=sampler,
        shuffle=True,
    )

    data_iterator = tqdm(
        static_dataloader,
        leave=True,
        unit="batch",
        postfix={
            "epo": -1,
            "acc": "%.4f" % 0.0,
            "lss": "%.8f" % 0.0,
            "dlb": "%.4f" % -1,
        },
        disable=silent,
    )

    kmeans = KMeans(n_clusters=model.cluster_number, n_init=20)
    model.train()
    features = []
    actual = []
    # from initial cluster centers
    for index, batch in enumerate(data_iterator):
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch        # if we have a prediction label, separate it to actual
            actual.append(value)
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(model.encoder(batch).detach().cpu())
    
    actual = torch.cat(actual).long()
    predicted = kmeans.fit_predict(torch.cat(features).numpy().squeeze(1))
    predicted_previous = torch.tensor(np.copy(predicted), dtype=torch.long)
    _, accuracy = cluster_accuracy(predicted, actual.cpu().numpy())
    cluster_centers = torch.tensor(
        kmeans.cluster_centers_, dtype=torch.float, requires_grad=True
    )
    if cuda:
        cluster_centers = cluster_centers.cuda(non_blocking=True)
    with torch.no_grad():
        # initialise the cluster centers
        model.state_dict()["assignment.cluster_centers"].copy_(cluster_centers)
    
    loss_function = nn.KLDivLoss(size_average=False)
    delta_label = None
    for epoch in range(epochs):
        features = []
        data_iterator = tqdm(train_dataloader, leave=True, unit='batch', 
                             postfix={
                                        "epo": epoch,
                                        "acc": "%.4f" % (accuracy or 0.0),
                                        "lss": "%.8f" % 0.0,
                                        "dlb": "%.4f" % (delta_label or 0.0),
                                        },
                             disable=silent,)
        model.train()
        for index, batch in enumerate(data_iterator):
            if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                batch, _ = batch  # if we have a prediction label, strip it away
            if cuda:
                batch = batch.cuda(non_blocking=True)
            output = model(batch).squeeze(1)
            target = target_distribution(output).detach()
            loss = loss_function(output.log(), target) / output.shape[0]
            data_iterator.set_postfix(
                epo=epoch,
                acc="%.4f" % (accuracy or 0.0),
                lss="%.8f" % float(loss.item()),
                dlb="%.4f" % (delta_label or 0.0),
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step(closure=None)
            features.append(model.encoder(batch).detach().cpu())
            if update_freq is not None and index % update_freq == 0:
                loss_value = float(loss.item())
                data_iterator.set_postfix(
                    epo=epoch,
                    acc="%.4f" % (accuracy or 0.0),
                    lss="%.8f" % loss_value,
                    dlb="%.4f" % (delta_label or 0.0),
                )
                if update_callback is not None:
                    update_callback(accuracy, loss_value, delta_label)
                
        predicted, actual = predict(
            dataset,
            model,
            batch_size=evaluate_batch_size,
            collate_fn=collate_fn,
            silent=True,
            return_actual=True,
            cuda=cuda,
        )
        delta_label = (
            float((predicted != predicted_previous).float().sum().item())
            / predicted_previous.shape[0]
        )
        if stopping_delta is not None and delta_label < stopping_delta:
            print(
                'Early stopping as label delta "%1.5f" less than "%1.5f".'
                % (delta_label, stopping_delta)
            )
            break
        predicted_previous = predicted
        _, accuracy = cluster_accuracy(predicted.cpu().numpy(), actual.cpu().numpy())
        data_iterator.set_postfix(
            epo=epoch,
            acc="%.4f" % (accuracy or 0.0),
            lss="%.8f" % 0.0,
            dlb="%.4f" % (delta_label or 0.0),
        )
        if epoch_callback is not None:
            epoch_callback(epoch, model)
    

def predict(
    dataset,
    model: torch.nn.Module,
    batch_size: int = 1024,
    collate_fn=default_collate,
    cuda: bool = True,
    silent: bool = False,
    return_actual: bool = False,
) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
    dataloader = DataLoader(
        dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False
    )
    data_iterator = tqdm(dataloader, leave=True, unit="batch", disable=silent,)
    features = []
    actual = []
    model.eval()
    for batch in data_iterator:
        if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
            batch, value = batch  # unpack if we have a prediction label
            if return_actual:
                actual.append(value)
        elif return_actual:
            raise ValueError(
                "Dataset has no actual value to unpack, but return_actual is set."
            )
        if cuda:
            batch = batch.cuda(non_blocking=True)
        features.append(
            model(batch).detach().cpu().squeeze(1)
        )  # move to the CPU to prevent out of memory on the GPU
    if return_actual:
        return torch.cat(features).max(1)[1], torch.cat(actual).long()
    else:
        return torch.cat(features.squeeze(1)).max(1)[1]


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

def target_distribution(batch: torch.Tensor) -> torch.Tensor:
    """
    Compute the target distribution p_ij, given the batch (q_ij), as in 3.1.3 Equation 3 of
    Xie/Girshick/Farhadi; this is used the KL-divergence loss function.

    :param batch: [batch size, number of clusters] Tensor of dtype float
    :return: [batch size, number of clusters] Tensor of dtype float
    """
    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()



if __name__ == '__main__':
    main()
