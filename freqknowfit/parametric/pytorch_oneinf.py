class OffsetLogitModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.offset = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
        # self.loss_fn = torch.nn.BCELoss()
        self.loss_fn = torch.nn.MSELoss()
        # self.loss_fn = torch.nn.L1Loss()

    def forward(self, x):
        lin = self.linear(x)
        # print("lin", lin)
        sig_pred = torch.sigmoid(lin)
        # return sig_pred
        sig_off = torch.sigmoid(self.offset)
        # print("sig", sig_pred, sig_off)
        # return sig_off + (1.0 - sig_off) * sig_pred
        return sig_off + sig_pred - sig_off * sig_pred

    def forward_loss(self, x, y):
        y_hat = self.forward(x)
        # print("y_hat", y_hat)
        loss = self.loss_fn(y_hat[:, 0], y)
        # print("loss", loss)
        return loss


def fit_offset_logit(x, y):
    from torch.optim import Adam

    model = OffsetLogitModel()
    """
    optimizer = LBFGS(
        ,
        lr=1,
        history_size=10,
        max_iter=300,
        tolerance_change=0,
        tolerance_grad=1e-5,
    )

    def closure():
        optimizer.zero_grad()
        loss = model.forward_loss(x, y)
        loss.backward()
        return loss

    optimizer.step(closure)
    state = optimizer.state[next(iter(optimizer.state))]
    print("state", state)
    """
    # x.requires_grad_(True)

    opt = Adam(
        [{"params": [model.offset], "lr": 0.1}, {"params": model.linear.parameters()}],
        1,
    )
    model.train()
    for epoch in range(100):
        # Train
        opt.zero_grad()
        loss = model.forward_loss(x, y)
        loss.backward()
        opt.step()
        train_loss = loss.item()
        print("loss", train_loss)

    return model
