


    # usage:
    self.jmp = skip_layer(dims, head, layer) # in your init
    x = self.jmp(x) # in your forward. Ideally upstream of your attention unless you want that attention to not be effected by the layer skipping.
     self.processor.jmp.update_threshold(loss=loss.item()) # in your forward model (or something similar) if you want the threshold for layer jumping to change based on loss -outside- the graph otherwise omit and it will adjust on gradients as usual
    
```python
class mgate(nn.Module):
    def __init__(self, dims, mem=64, thresh=0.5):
        super().__init__()
        self.mkey = nn.Parameter(torch.randn(mem, dims))
        self.mval = nn.Parameter(torch.randn(mem, 1))
        self.mlp = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        self.threshold = nn.Parameter(torch.tensor(thresh, dtype=torch.float32), requires_grad=False)
        self.concat = nn.Linear(2,1, device=device, dtype=dtype)

    def forward(self, x):
        key = F.softmax(torch.matmul(F.normalize(x, p=2, dim=-1), F.normalize(self.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x.shape[-1]), dim=-1)
        x = self.concat(torch.cat((torch.matmul(key, self.mval),  self.mlp(x)), dim=-1))
       
        threshold = apply_ste_threshold(x, self.threshold)
        return threshold, x

class StraightThroughThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold):
        binary_output = (x > threshold).float()
        ctx.save_for_backward(x)
        return binary_output

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad_x = grad_output.clone()
        grad_threshold = None
        return grad_x, grad_threshold

apply_ste_threshold = StraightThroughThreshold.apply

class skip_layer(nn.Module):
    def __init__(self, dims, head, layer):
        super().__init__()

        self.work_mem = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        self.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)
        self.layer = layer
        self.loss = 0
  
        self.layers = nn.ModuleList()
        for i in range(layer):
            self.layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
                'mgate': mgate(dims, mem=64),
                }))

        self.mgate= mgate(dims, mem=64)
        self.policy_net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3))

        self.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(dims, dims * 4), nn.GELU(), nn.Linear(dims * 4, dims))
        self.mlp_ln = nn.LayerNorm(dims)
 
    def update_threshold(self, loss, lr=0.01):
        if loss > self.loss:
            self.mgate.threshold.sub_(lr)
        else:
           self.mgate.threshold.add_(lr)
        self.mgate.threshold.data = torch.clamp(self.mgate.threshold.data, 0.0, 1.0)

    def forward(self, x, xa=None): 
        batch, ctx = x.shape[:2]
        ox = x
        work_mem = self.work_mem.expand(batch, -1, -1)
        x1 = x.mean(dim=1)

        policy_logits = self.policy_net(x1)
        policy = F.softmax(policy_logits, dim=-1)
        
        history = []
        i = 0
        while i < self.layer:
            layer = self.layers[i]
            # scalar = self.layer_choice(x, i)
            scalar, choice = layer['mgate'](x)
            mask = scalar.expand(-1, ctx, -1) 
            x2 = torch.zeros_like(x)
            skip = (scalar == 0).squeeze(-1)
            x2[skip] = x[skip].clone().detach() 

            px = layer['ln'](x2)  
            if layer['adapter'] is not None:
                attn = layer['adapter'](px)
            gate_val = layer['gate'](px)
            x = x + gate_val * (attn * mask)

            mem = x.mean(dim=1, keepdim=True)
            mem_val = self.mem_gate(mem)
            work_mem = mem_val * work_mem + (1 - mem_val) * mem
            if i < self.layer - 1:
                action = torch.multinomial(policy, 1).squeeze(1).item()
            else:
                action = 0
            distance = 0
            if action == 1: distance = 1
            if action == 2: distance = 2
            if distance > 0:
                i_next = min(i + distance, self.layer - 1)
                jump = self.jump_weights[min(distance-1, 2)]               
                x = x + jump * ox + (1-jump) * work_mem.expand(-1, ctx, -1)
                i = i_next
                history.append(i)
            else:
                i += 1
        
        x3 = self.mlp_gate(x)
        output = self.mlp(self.mlp_ln(x))


```
        x = x + x3 * output
        self.logs = {'jumps': history}
        return x
