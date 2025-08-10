
class skip_layer(nn.Module):
    def __init__(self, dims, head, layer, threshold=0.1):
        super().__init__()

        self.threshold = nn.Parameter(torch.tensor(threshold, device=device, dtype=dtype), requires_grad=True)
        self.work_mem = nn.Parameter(torch.zeros(1, 1, dims), requires_grad=True)
        self.mem_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]), requires_grad=True)

        self.dims = dims
        self.head = head
        self.head_dim = dims // head
        self.layer = layer

        self.attention = attentiona(dims, head)
        
        self.predict = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dims),
                nn.Linear(dims, 1),
                nn.Sigmoid()
            ) for _ in range(layer)])

        self.layers = nn.ModuleList()
        for i in range(layer):
            self.layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None}))
        
        self.policy_net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3))

        self.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(dims, dims * 4), nn.GELU(), nn.Linear(dims * 4, dims))
        self.mlp_ln =nn.LayerNorm(dims)

    def _calculate_shared_attention(self, x, mask=None):
        return self.attention(x, xa=x, mask=None)

    def predict_choice(self, x, layer_idx):
        choice = self.predict[layer_idx](x)
        return (choice > self.threshold).float()
    
    def forward(self, x, xa=None, mask=None):
        batch, ctx = x.shape[:2]
        work_mem = self.work_mem.expand(batch, -1, -1)
        og_x = x
        pooled_rep = x.mean(dim=1)
        policy_logits = self.policy_net(pooled_rep)
        policy = F.softmax(policy_logits, dim=-1)
        
        history = []
        i = 0
        while i < self.layer:
            layer = self.layers[i]
            choice = self.predict_choice(x, i)
            if choice.mean() < 0.2 and i > 0:
                i += 1
                history.append(i)
                continue
                
            x = layer['ln'](x)
            mask_base = choice.unsqueeze(1).contiguous()
            combined_mask = None
            if mask is None:
                combined_mask = mask_base 
            else:
                combined_mask = mask.contiguous() * mask_base
                
            if choice.mean() > 0.3:
                attn_output = self._calculate_shared_attention(x, mask=combined_mask.contiguous())
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                
                gate_value = layer['gate'](x)
                x = x + gate_value * attn_output
                mem_gate = self.mem_gate(x)
                work_mem = mem_gate * work_mem + (1 - mem_gate) * x.mean(dim=1, keepdim=True)
            
            jump_prob = policy[:, 1] if i < self.layer - 1 else torch.zeros_like(policy[:, 1])
            jump = (torch.rand_like(jump_prob) < jump_prob).any()
            
            if jump:
                distance = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                i_next = min(i + distance, self.layer - 1)
                skip_weight = self.jump_weights[min(distance-1, 2)]
                x = x + skip_weight * og_x + (1-skip_weight) * work_mem
                i = i_next
                history.append(i)
            else:
                i += 1
        
        mlp_choice = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_choice * mlp_output
        return x, {'jumps': history}
        
