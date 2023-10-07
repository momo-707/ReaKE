import tmap as tm
from faerun import Faerun
import torch
import pickle
from dgl.dataloading import GraphDataLoader
from dgl.data.utils import load_info
from model import Reaction_KG


def draw(args, data):
    #加载模型
    path = '../saved/' + args.pretrained_model + '/'
    path_parameter = path + 'parameter_info.pkl'
    feature_len = load_info(path_parameter)['feature_length']

    print('loading hyperparameters of pretrained model from ' + path + 'args.pkl')
    with open(path + 'args.pkl', 'rb') as f:
        hparams = pickle.load(f)

    print('loading pretrained model from ' + path + 'model.pt')
    mole = Reaction_KG(hparams, feature_len)


    if torch.cuda.is_available():
        mole.load_state_dict(torch.load(path + 'model.pt'))
        mole = mole.cuda(args.gpu)
    else:
        mole.load_state_dict(torch.load(path + 'model.pt', map_location=torch.device('cpu')))

    train_data, val_data, test_data = data
    test_data = data
    test_dataloader = GraphDataLoader(test_data, batch_size=args.batch_size)
    train_dataloader = GraphDataLoader(train_data, batch_size=args.batch_size, shuffle=False)
    val_dataloader = GraphDataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    features, types = [], []


    with torch.no_grad():
        mole.eval()
        for reactant, prod, labels in train_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = reactant_embeddings - prod_embeddings  # 128*1024
            features.append(embeddings)
            types.append(labels)
        for reactant, prod, labels in val_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = reactant_embeddings - prod_embeddings  # 128*1024
            features.append(embeddings)
            types.append(labels)
        for reactant, prod, labels in test_dataloader:
            reactant_embeddings = mole(h=reactant, mode='downstream')
            prod_embeddings = mole(h=prod, mode='downstream')
            embeddings = reactant_embeddings - prod_embeddings  # 128*1024
            features.append(embeddings)
            types.append(labels)
        features = torch.cat(features, dim=0).cpu().numpy()
        types = torch.cat(types, dim=0).cpu().numpy()

    dims = 1024
    enc = tm.Minhash(dims)
    lf = tm.LSHForest(dims, 128, weighted=True)

    #features = tm.VectorFloat(list(features))
    print("Running tmap ...")

    lf.batch_add(enc.batch_from_weight_array(features))
    lf.index()
    cfg = tm.LayoutConfiguration()
    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)

    faerun = Faerun(clear_color="#111111", view="front", coords=False)
    faerun.add_scatter(
        "reake",
        {"x": x, "y": y, "c": types},
        colormap=faerun.discrete_cmap(46, "Spectral"),
        shader="smoothCircle",
        point_scale=2.5,
        max_point_size=10,
        has_legend=True,
        categorical=True,
    )
    faerun.add_tree(
        "reake_tree", {"from": s, "to": t}, point_helper="reake", color="#666666"
    )
    faerun.plot("reake", template="url_image")