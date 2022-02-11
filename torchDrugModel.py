import torch, os

from torch.utils import data as torch_data
from torchdrug import core, datasets, tasks, models, data, utils


class DRKG(data.KnowledgeGraphDataset):
    """
    DRKG for knowledge graph reasoning.

    Parameters:
        path (str): path to store the dataset
        verbose (int, optional): output verbose level
    """

    url = "https://dgl-data.s3-us-west-2.amazonaws.com/dataset/DRKG/drkg.tar.gz"

    md5 = "40519020c906ffa9c821fa53cd294a76"

    def __init__(self, path, verbose = 1):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path

        zip_file = utils.download(self.url, path, md5 = self.md5)
        tsv_file = utils.extract(zip_file, "drkg.tsv")

        self.load_tsv(tsv_file, verbose = verbose)


if __name__ == '__main__':
    dataset = DRKG("./data")

    lengths = [int(0.8 * len(dataset)), int(0.1 * len(dataset))]
    lengths += [len(dataset) - sum(lengths)]
    train_set, valid_set, test_set = torch_data.random_split(dataset, lengths)

    print("train: ", len(train_set), "val: ", len(valid_set), "test: ", len(test_set))

    model = models.RotatE(num_entity = dataset.num_entity,
                          num_relation = dataset.num_relation,
                          embedding_dim = 2048, max_score = 9)

    task = tasks.KnowledgeGraphCompletion(model, num_negative = 256,
                                          adversarial_temperature = 1)

    optimizer = torch.optim.Adam(task.parameters(), lr = 2e-5)
    solver = core.Engine(task, train_set, valid_set, test_set, optimizer,
                         batch_size = 1024)
    solver.train(num_epoch = 1)
    solver.evaluate("valid")

    triples = test_set[:10]
    preds = task.cpu().predict(triples)
    tails = torch.softmax(preds[:, 0, :].detach(), dim = 1)
    heads = torch.softmax(preds[:, 1, :].detach(), dim = 1)
    top_tails = tails.topk(1)
    top_heads = heads.topk(1)
    for i, [h, t, r] in enumerate(triples):
        print("(%s)-[%s]->(%s): %s " % (h.item(), r.item(), top_tails.indices[i].item(), "{:.2%}".format(top_tails.values[i].item())))