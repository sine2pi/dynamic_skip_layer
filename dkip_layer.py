

class MgateDecision(nn.Module):
    def __init__(self, dims, mem=64):
        super().__init__()
        self.mkey = nn.Parameter(torch.randn(mem, dims))
        self.mval = nn.Parameter(torch.randn(mem, 1))
        self.mlp = nn.Sequential(nn.Linear(dims, dims//2), nn.SiLU(), nn.Linear(dims//2, 1))
        self.threshold = nn.Parameter(torch.tensor(0.5, dtype=torch.float32), requires_grad=True)
        self.combiner = nn.Linear(2,1, device=device, dtype=dtype)

    def forward(self, x1, x2, cos=False):
        x3 = torch.cat((x1, x2), dim=-1)
        if cos:
            key = F.softmax(torch.matmul(F.normalize(x3, p=2, dim=-1), F.normalize(self.mkey, p=2, dim=-1).transpose(0, 1)) / math.sqrt(x3.shape[-1]), dim=-1)
        else:
            key = F.softmax(torch.matmul(x3, self.mkey.transpose(0, 1)) / math.sqrt(x3.shape[-1]), dim=-1)
        choice = self.combiner(torch.cat((torch.matmul(key, self.mval),  self.mlp(x3)), dim=-1))
        decision = apply_ste_threshold(choice, self.threshold)
        return decision, choice 

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
    def __init__(self, dims, head, num_layers, initial_threshold=0.5, ema_decay=0.99, threshold_lr=0.01):
        super().__init__()

        self.logs = []

        self.initial_threshold = initial_threshold
        self.threshold = nn.Parameter(torch.tensor(initial_threshold, dtype=torch.float32), requires_grad=False) 
        
        self.ema_loss = torch.tensor(0.0, dtype=torch.float32)
        self.ema_decay = ema_decay
        self.threshold_lr = threshold_lr

        self.work_mem = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        self.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)

        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.num_layers = num_layers

        self.attention = attention(dims, head)
        
        self.predict = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dims),
                nn.Linear(dims, 1),
                nn.Sigmoid() 
            ) for _ in range(num_layers)])

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None,
                'mgate_decision': MgateDecision(dims=dims * 2, mem=64),
                }))
        
        self.policy_net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3))

        self.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(dims, dims * 4), nn.GELU(), nn.Linear(dims * 4, dims))
        self.mlp_ln =nn.LayerNorm(dims)

    def _calculate_shared_attention(self, x, mask=None):
        return self.attention(x, xa=x, mask=mask)

    def predict_choice(self, pooled_rep, layer_idx):
        choice_scores = self.predict[layer_idx](pooled_rep)
        return choice_scores

    def update_threshold(self, current_batch_loss):
        with torch.no_grad():
            if self.ema_loss.item() == 0.0:
                self.ema_loss.copy_(current_batch_loss)
            else:
                self.ema_loss.mul_(self.ema_decay).add_(current_batch_loss * (1 - self.ema_decay))
            if current_batch_loss > self.ema_loss:
                self.threshold.sub_(self.threshold_lr)
            else:
                self.threshold.add_(self.threshold_lr)
            self.threshold.data = torch.clamp(self.threshold.data, 0.0, 1.0)

    def forward(self, x, xa=None, mask=None):
        batch, ctx, _ = x.shape
        ox = x
        work_mem = self.work_mem.expand(batch, -1, -1)
        choose = x.mean(dim=1)
        policy = F.softmax(self.policy_net(choose), dim=-1)
        
        history = []
        chose = torch.zeros_like(choose)
        
        i = 0
        while i < self.num_layers:
            layer = self.layers[i]
            mask_scalar, choice = layer['mgate_decision'](choose, chose)
            mask_tokens = mask_scalar.unsqueeze(-1).expand(-1, ctx, -1) 
            
            is_skipped = (mask_scalar == 0).squeeze(-1)
            chose[is_skipped] = choose[is_skipped].clone().detach() 
            
            x_norm = layer['ln'](x)
            attn_output = self._calculate_shared_attention(x_norm, mask)

            if layer['adapter'] is not None:
                attn_output = layer['adapter'](attn_output)
            
            gate_val = layer['gate'](x)
            x = x + gate_val * (attn_output * mask_tokens)

            to_mem = x.mean(dim=1, keepdim=True)
            mem_val = self.mem_gate(to_mem)
            work_mem = mem_val * work_mem + (1 - mem_val) * to_mem
            
            if i < self.num_layers - 1:
                action = torch.multinomial(policy, 1).squeeze(1).item()
            else:
                action = 0

            jump_distance = 0
            if action == 1: jump_distance = 1
            if action == 2: jump_distance = 2

            if jump_distance > 0:
                i_next = min(i + jump_distance, self.num_layers - 1)
                skip_weight = self.jump_weights[min(jump_distance-1, 2)]               
                x = x + skip_weight * ox + (1-skip_weight) * work_mem.expand(-1, ctx, -1)
                
                i = i_next
                history.append(i)
            else:
                i += 1
        
        choice = self.mlp_gate(x)
        output = self.mlp(self.mlp_ln(x))
        x = x + choice * output
        self.logs = {'jumps': history}
        return x

