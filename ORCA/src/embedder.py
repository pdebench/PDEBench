import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from timeit import default_timer
from functools import partial
from transformers import AutoModel, AutoConfig, SwinForImageClassification, SwinForMaskedImageModeling, RobertaForTokenClassification
from otdd.pytorch.distance import DatasetDistance, FeatureCost

from task_configs import get_data, get_optimizer_scheduler
from utils import conv_init, embedder_init, embedder_placeholder, adaptive_pooler, to_2tuple, set_grad_state, create_position_ids_from_inputs_embeds, l2, MMD_loss
import copy


def otdd(feats, ys=None, src_train_dataset=None, exact=True):
    ys = torch.zeros(len(feats)) if ys is None else ys

    if not torch.is_tensor(feats):
        feats = torch.from_numpy(feats).to('cpu')
        ys = torch.from_numpy(ys).long().to('cpu')

    dataset = torch.utils.data.TensorDataset(feats, ys)

    dist = DatasetDistance(src_train_dataset, dataset,
                                    inner_ot_method = 'exact' if exact else 'gaussian_approx',
                                    debiased_loss = True, inner_ot_debiased=True,
                                    p = 2, inner_ot_p=2, entreg = 1e-1, ignore_target_labels = False,
                                    device=feats.device, load_prev_dyy1=None)
                
    d = dist.distance(maxsamples = len(src_train_dataset))
    return d


class wrapper2D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='base', train_epoch=0, activation=None, target_seq_len=None, drop_out=None, from_scratch=False):
        super().__init__()
        self.classification = (not isinstance(output_shape, tuple)) and (output_shape != 1)
        self.output_raw = True

        if weight == 'tiny':
            arch_name = "microsoft/swin-tiny-patch4-window7-224"
            embed_dim = 96
            output_dim = 768
            img_size = 224
        elif weight == 'base':
            arch_name = "microsoft/swin-base-patch4-window7-224-in22k"
            embed_dim = 128
            output_dim = 1024
            img_size = 224
            patch_size = 4

        if self.classification:
            modelclass = SwinForImageClassification
        else:
            modelclass = SwinForMaskedImageModeling
            
        self.model = modelclass.from_pretrained(arch_name)
        self.model.config.image_size = img_size
        if drop_out is not None:
            self.model.config.hidden_dropout_prob = drop_out 
            self.model.config.attention_probs_dropout_prob = drop_out

        self.model = modelclass.from_pretrained(arch_name, config=self.model.config) if not from_scratch else modelclass(self.model.config)

        if self.classification:
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity()
            self.predictor = nn.Linear(in_features=output_dim, out_features=output_shape)
        else:
            self.pool_seq_dim = adaptive_pooler(output_shape[1] if isinstance(output_shape, tuple) else 1)
            self.pool = nn.AdaptiveAvgPool2d(input_shape[-2:])
            self.predictor = nn.Sequential(self.pool_seq_dim, self.pool)

        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)

        if use_embedder:
            self.embedder = Embeddings2D(input_shape, patch_size=patch_size, config=self.model.config, embed_dim=embed_dim, img_size=img_size)
            embedder_init(self.model.swin.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)
            self.model.swin.embeddings = self.embedder  


    def forward(self, x):
        if self.output_raw:
            return self.model.swin.embeddings(x)[0]
            
        x = self.model(x).logits

        return self.predictor(x)


class wrapper1D(torch.nn.Module):
    def __init__(self, input_shape, output_shape, use_embedder=True, weight='roberta', train_epoch=0, activation=None, target_seq_len=512, drop_out=None, from_scratch=False):
        super().__init__()

        self.dense = False
        self.output_raw = True
        self.weight = weight
        self.output_shape = output_shape

        if isinstance(output_shape, tuple):
            self.dense = True

        if weight =='swin':
            self.model = SwinForImageClassification.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k") if not from_scratch else SwinForImageClassification()
            self.model.pooler = nn.AdaptiveAvgPool1d(1)
            self.model.classifier = nn.Identity() 

        else:
            modelname = 'roberta-base' if weight[:7] == 'roberta' else 'bert-base-uncased'
            configuration = AutoConfig.from_pretrained(modelname)
            if drop_out is not None:
                configuration.hidden_dropout_prob = drop_out
                configuration.attention_probs_dropout_prob = drop_out
            self.model = AutoModel.from_pretrained(modelname, config = configuration) if not from_scratch else AutoModel.from_config(configuration)

        if use_embedder:
            self.embedder = Embeddings1D(input_shape, config=self.model.config, embed_dim=128 if weight == 'swin' else 768, target_seq_len=1024 if weight == 'swin' else target_seq_len, dense=self.dense)
            embedder_init(self.model.swin.embeddings if weight == 'swin' else self.model.embeddings, self.embedder, train_embedder=train_epoch > 0)
            set_grad_state(self.embedder, True)    
        else:
            self.embedder = nn.Identity()

        if not weight == 'swin': 
            self.model.embeddings = embedder_placeholder()
            if self.dense:
                self.model.pooler = nn.Identity()
                self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num, output_shape=output_shape, dense=True)
            else:
                self.model.pooler = adaptive_pooler()
                self.predictor = nn.Linear(in_features=768, out_features=output_shape)   
        else:
            self.model.swin.embeddings = self.embedder  
            if self.dense:
                self.predictor = adaptive_pooler(out_channel = output_shape[-2] * self.embedder.stack_num)
            else:
                self.predictor = nn.Linear(in_features=1024, out_features=output_shape)  

        if activation == 'sigmoid':
            self.predictor = nn.Sequential(self.predictor, nn.Sigmoid())  
            
        set_grad_state(self.model, False)
        set_grad_state(self.predictor, False)


    def forward(self, x):
        if self.weight == 'swin':
            if self.output_raw:
                return self.model.swin.embeddings(x)[0]

            x = self.model(x).logits
            return self.predictor(x)

        if self.output_raw:
            return self.embedder(x) 

        x = self.embedder(x)

        if self.dense:
            x = self.model(inputs_embeds=x)['last_hidden_state']
            x = self.predictor(x)
        else:
            x = self.model(inputs_embeds=x)['pooler_output']
            x = self.predictor(x)

        if x.shape[1] == 1 and len(x.shape) == 2:
            x = x.squeeze(1)

        return x


class Embeddings2D(nn.Module):

    def __init__(self, input_shape, patch_size=4, embed_dim=96, img_size=224, config=None):
        super().__init__()

        self.resize, self.input_dimensions = transforms.Resize((img_size, img_size)), (img_size, img_size)
        self.patch_size = to_2tuple(patch_size)
        self.patched_dimensions = (self.input_dimensions[0] // self.patch_size[0], self.input_dimensions[1] // self.patch_size[1])
        ks = self.patch_size
        self.projection = nn.Conv2d(input_shape[1], embed_dim, kernel_size=ks, stride=self.patch_size, padding=(ks[0]-self.patch_size[0]) // 2)
        self.norm = nn.LayerNorm(embed_dim)
        num_patches = (self.input_dimensions[1] // self.patch_size[1]) * (self.input_dimensions[0] // self.patch_size[0])
        
        conv_init(self.projection)

        
    def maybe_pad(self, x, height, width):
        if width % self.patch_size[1] != 0:
            pad_values = (0, self.patch_size[1] - width % self.patch_size[1])
            x = nn.functional.pad(x, pad_values)
        if height % self.patch_size[0] != 0:
            pad_values = (0, 0, 0, self.patch_size[0] - height % self.patch_size[0])
            x = nn.functional.pad(x, pad_values)
        return x


    def forward(self, x, *args, **kwargs):
        x = self.resize(x)
        _, _, height, width = x.shape

        x = self.maybe_pad(x, height, width)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        
        x = self.norm(x)   
        
        return x, self.patched_dimensions


class Embeddings1D(nn.Module):
    def __init__(self, input_shape, embed_dim=768, target_seq_len=64, config=None, dense=False):
        super().__init__()
        self.dense = dense
        self.embed_dim = embed_dim
        self.stack_num = self.get_stack_num(input_shape[-1], target_seq_len)
        self.patched_dimensions = (int(np.sqrt(input_shape[-1] // self.stack_num)), int(np.sqrt(input_shape[-1] // self.stack_num)))
        self.norm = nn.LayerNorm(embed_dim)
        self.padding_idx = 1
        self.position_embeddings = nn.Embedding(target_seq_len, embed_dim, padding_idx=self.padding_idx)

        self.projection = nn.Conv1d(input_shape[1], embed_dim, kernel_size=self.stack_num, stride=self.stack_num)
        conv_init(self.projection)


    def get_stack_num(self, input_len, target_seq_len):
        if self.embed_dim == 768:
            for i in range(1, input_len + 1):
                if input_len % i == 0 and input_len // i <= target_seq_len:
                    break
            return i
        else:
            for i in range(1, input_len + 1):
                root = np.sqrt(input_len // i)
                if input_len % i == 0 and input_len // i <= target_seq_len and int(root + 0.5) ** 2 == (input_len // i):
                    break
            return i


    def forward(self, x=None, inputs_embeds=None, *args, **kwargs):
        if x is None:
            x = inputs_embeds
        b, c, l = x.shape

        x = self.projection(x).transpose(1, 2)
        x = self.norm(x)
            
        position_ids = create_position_ids_from_inputs_embeds(x, self.padding_idx)
        self.ps = self.position_embeddings(position_ids)
        x = x + self.ps

        if self.embed_dim == 768:
            return x
        else:
            return x, self.patched_dimensions



####################################################

def get_tgt_model(args, root, sample_shape, num_classes, loss, add_loss=False, use_determined=False, context=None, opid=0):
    
    src_train_loader, _, _, _, _, _, _ = get_data(root, args.embedder_dataset, args.batch_size, False, maxsize=5000)
    if len(sample_shape) == 4:
        IMG_SIZE = 224 if args.weight == 'tiny' or args.weight == 'base' else 196
            
        src_model = wrapper2D(sample_shape, num_classes, use_embedder=False, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, drop_out=args.drop_out)
        src_model = src_model.to(args.device).eval()
            
        src_feats = []
        src_ys = []
        for i, data in enumerate(src_train_loader):
            x_, y_ = data 
            x_ = x_.to(args.device)
            x_ = transforms.Resize((IMG_SIZE, IMG_SIZE))(x_)
            out = src_model(x_)
            if len(out.shape) > 2:
                out = out.mean(1)

            src_ys.append(y_.detach().cpu())
            src_feats.append(out.detach().cpu())
        src_feats = torch.cat(src_feats, 0)
        src_ys = torch.cat(src_ys, 0).long()
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)        
        del src_model    

    else:
        src_feats, src_ys = src_train_loader.dataset.tensors[0].mean(1), src_train_loader.dataset.tensors[1]
        src_train_dataset = torch.utils.data.TensorDataset(src_feats, src_ys)
        
    tgt_train_loader, _, _, n_train, _, _, data_kwargs = get_data(root, args.dataset, args.batch_size, False, get_shape=True)
    transform = data_kwargs['transform'] if data_kwargs is not None and 'transform' in data_kwargs else None
        
    if args.infer_label:
        tgt_train_loader, num_classes_new = infer_labels(tgt_train_loader)
    else:
        num_classes_new = num_classes

    print("src feat shape", src_feats.shape, src_ys.shape, "num classes", num_classes_new) 

    tgt_train_loaders, tgt_class_weights = load_by_class(tgt_train_loader, num_classes_new)

    wrapper_func = wrapper1D if len(sample_shape) == 3 else wrapper2D
    tgt_model = wrapper_func(sample_shape, num_classes, weight=args.weight, train_epoch=args.embedder_epochs, activation=args.activation, target_seq_len=args.target_seq_len, drop_out=args.drop_out)
    tgt_model = tgt_model.to(args.device).train()

    args, tgt_model, tgt_model_optimizer, tgt_model_scheduler = get_optimizer_scheduler(args, tgt_model, module='embedder')
    tgt_model_optimizer.zero_grad()

    if args.objective == 'otdd-exact':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=True)
    elif args.objective == 'otdd-gaussian':
        score_func = partial(otdd, src_train_dataset=src_train_dataset, exact=False)
    elif args.objective == 'l2':
        score_func = partial(l2, src_train_dataset=src_train_dataset)
    else:
        score_func = MMD_loss(src_data=src_feats, maxsamples=args.maxsamples)
    
    score = 0
    total_losses, times, embedder_stats = [], [], []
    
    for ep in range(args.embedder_epochs):   

        total_loss = 0    
        time_start = default_timer()

        for i in np.random.permutation(num_classes_new):
            feats = []
            datanum = 0

            for j, data in enumerate(tgt_train_loaders[i]):
                
                if transform is not None:
                    x, y, z = data
                else:
                    x, y = data 
                
                x = x.to(args.device)
                out = tgt_model(x)
                feats.append(out)
                datanum += x.shape[0]
                
                if datanum > args.maxsamples: break

            feats = torch.cat(feats, 0).mean(1)
            if feats.shape[0] > 1:
                loss = tgt_class_weights[i] * score_func(feats)
                loss.backward()
                total_loss += loss.item()

        time_end = default_timer()  
        times.append(time_end - time_start) 

        total_losses.append(total_loss)
        embedder_stats.append([total_losses[-1], times[-1]])
        print("[train embedder", ep, "%.6f" % tgt_model_optimizer.param_groups[0]['lr'], "] time elapsed:", "%.4f" % (times[-1]), "\totdd loss:", "%.4f" % total_losses[-1])

        tgt_model_optimizer.step()
        tgt_model_scheduler.step()
        tgt_model_optimizer.zero_grad()

    del tgt_train_loader, tgt_train_loaders
    torch.cuda.empty_cache()

    tgt_model.output_raw = False

    return tgt_model, embedder_stats


def infer_labels(loader, k = 10):
    from sklearn.cluster import k_means, MiniBatchKMeans
    
    if hasattr(loader.dataset, 'tensors'):
        X, Y = loader.dataset.tensors[0].cpu(), loader.dataset.tensors[1].cpu().numpy()
        try:
            Z = loader.dataset.tensors[2].cpu()
        except:
            Z = None
    else:
        X, Y, Z = get_tensors(loader.dataset)

    Y = Y.reshape(len(Y), -1)

    if len(Y) <= 10000:
        labeling_fun = lambda Y: torch.LongTensor(k_means(Y, k)[1])
        Y = labeling_fun(Y).unsqueeze(1)
    else:
        kmeans = MiniBatchKMeans(n_clusters=k, batch_size=10000).fit(Y)
        Y = torch.LongTensor(kmeans.predict(Y)).unsqueeze(1)

    if Z is None:
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k
    return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, Y, Z), batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True), k


def load_by_class(loader, num_classes):
    train_set = loader.dataset
    subsets = {}

    if len(train_set.__getitem__(0)) == 3:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y, _) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    else:
        try:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y == target]) for target in range(num_classes)}
        except:
            subsets = {target: torch.utils.data.Subset(train_set, [i for i, (x, y) in enumerate(train_set) if y.item() == target]) for target in range(num_classes)}
    loaders = {target: torch.utils.data.DataLoader(subset, batch_size=loader.batch_size, shuffle=True, num_workers=4, pin_memory=True) for target, subset in subsets.items()}
    class_weights = {target: len(subset)/len(train_set) for target, subset in subsets.items()}
    
    print("class weights")
    for target, subset in subsets.items():
        print(target, len(subset), len(train_set), len(subset)/len(train_set))

    return loaders, class_weights



def get_tensors(dataset):
    xs, ys, zs = [], [], []
    for i in range(dataset.__len__()):
        data = dataset.__getitem__(i)
        xs.append(np.expand_dims(data[0], 0))
        ys.append(np.expand_dims(data[1], 0))
        if len(data) == 3:
            zs.append(np.expand_dims(data[2], 0))

    xs = torch.from_numpy(np.array(xs)).squeeze(1)
    ys = torch.from_numpy(np.array(ys)).squeeze(1)

    if len(zs) > 0:
        zs = torch.from_numpy(np.array(zs)).squeeze(1)
    else:
        zs = None

    return xs, ys, zs