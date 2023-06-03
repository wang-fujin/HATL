import torch
import torch.nn as nn
from Net import ResNet1d, Predictor
from Loss import MMDLoss, ACLoss

class HTLNet(nn.Module):
    def __init__(self,args):
        super(HTLNet,self).__init__()
        self.args = args
        self.backbone = ResNet1d(input_channel=args.input_channel,embedding_length=args.embedding_length)
        self.predictor = Predictor(embedding_length=args.embedding_length)

        self.predict_loss = nn.MSELoss()
        self.domain_loss = MMDLoss()
        self.ca_loss = ACLoss(loss_type=args.ca_loss_type)

        if args.use_dsbn:
            self.source_domain_label = 'source'
            self.target_domain_label = 'target'
        else:
            self.source_domain_label = 'target'
            self.target_domain_label = 'target'

    def forward(self,source,target,source_label):
        source_embedding, source_out, source_ca = self.backbone(source,domain_label=self.source_domain_label)
        target_embedding, target_out, target_ca = self.backbone(target,domain_label=self.target_domain_label)

        domain_loss = self.domain_loss(source_embedding,target_embedding)

        pred_label = self.predictor(source_embedding)
        pred_loss = self.predict_loss(pred_label,source_label)


        mmd_loss = self.domain_loss(source_out[0],target_out[0]) + self.domain_loss(source_out[1],target_out[1])

        ca_loss = self.ca_loss(source_ca[-1],target_ca[-1]) + self.ca_loss(source_ca[-2], target_ca[-2])

        return pred_loss, domain_loss, ca_loss, mmd_loss

    def predict(self,x,domain_label='target'):
        target_embedding,target_out, target_ca = self.backbone(x, domain_label=domain_label)
        pred_label = self.predictor(target_embedding)

        return pred_label

    def get_parameters(self,initial_lr=1.0):
        params = [
            {'params': self.predictor.parameters(), 'lr': initial_lr},
            {'params': self.backbone.parameters(), 'lr': initial_lr},
        ]
        return params

    def get_embedding(self, x, domain_label='target'):
        embedding, out, ca = self.backbone(x, domain_label=domain_label)
        return embedding


if __name__ == '__main__':
    from Config import get_args
    args = get_args()


    model = HTLNet(args)


    xs = torch.rand((args.batch_size,args.input_channel,128))
    xt = torch.rand((args.batch_size,args.input_channel,128))
    label = torch.rand((args.batch_size,1))


    pred_loss, domain_loss, ca_loss, mmd_loss = model(xs,xt,label)
    print('loss:',domain_loss,pred_loss,ca_loss)

    embedding = model.get_embedding(xs)
    print('embedding shape:',embedding.shape)

    pred = model.predict(xs)
    print('pred shape:',pred.shape)

