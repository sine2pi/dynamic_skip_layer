
class skip_layer(nn.Module): # Dynamic Skip Layer (DSL)
    def __init__(self, dims, head, layer, threshold=0.1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layer = layer

        self.threshold = threshold
        self.dims = dims
        self.head = head
        self.head_dim = dims // head

        self.attention_module = attentiona(dims, head) # use whatever attention 
      
        self.node_predictors = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dims),
                nn.Linear(dims, 1),
                nn.Sigmoid()
            ) for _ in range(layer)
        ])
        
        for i in range(layer):
            self.layers.append(nn.ModuleDict({
                'ln': nn.LayerNorm(dims),
                'gate': nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid()),
                'adapter': nn.Linear(dims, dims) if i % 2 == 0 else None
            }))
        
        self.policy_net = nn.Sequential(
            nn.Linear(dims, 128),
            nn.ReLU(),
            nn.Linear(128, 3))
        
        self.jump_weights = nn.Parameter(torch.tensor([0.1, 0.05, 0.01]))
        
        n_mlp = dims * 4
        self.mlp_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())
        self.mlp = nn.Sequential(nn.Linear(dims, n_mlp), nn.GELU(), nn.Linear(n_mlp, dims))
        self.mlp_ln =nn.LayerNorm(dims)
        self.working_memory = nn.Parameter(torch.zeros(1, 1, dims))
        self.memory_gate = nn.Sequential(nn.Linear(dims, 1), nn.Sigmoid())

    def _calculate_shared_attention(self, x, mask=None):
        return self.attention_module(x, xa=x, mask=None) # use whatever multiples of attentions for routing 

    def predict_node_importance(self, x, layer_idx):
        importance = self.node_predictors[layer_idx](x)
        return (importance > self.threshold).float()
    
    def forward(self, x, xa=None, mask=None):
        batch, ctx = x.shape[:2]

        working_memory = self.working_memory.expand(batch, -1, -1)
        original_x = x
        pooled_representation = x.mean(dim=1)
        policy_logits = self.policy_net(pooled_representation)
        policy = F.softmax(policy_logits, dim=-1)
        
        jump_history = []
        i = 0
        while i < self.layer:
            layer = self.layers[i]
            node_importance = self.predict_node_importance(x, i)
            if node_importance.mean() < 0.2 and i > 0:
                i += 1
                jump_history.append(i)
                continue
                
            norm_x = layer['ln'](x)
            importance_mask_base = node_importance.unsqueeze(1).contiguous()
            combined_custom_mask = None
            if mask is None:
                combined_custom_mask = importance_mask_base 
            else:
                combined_custom_mask = mask.contiguous() * importance_mask_base
                
            if node_importance.mean() > 0.3:
                attn_output = self._calculate_shared_attention(norm_x, mask=combined_custom_mask.contiguous())
                if layer['adapter'] is not None:
                    attn_output = layer['adapter'](attn_output)
                
                gate_value = layer['gate'](norm_x)
                x = x + gate_value * attn_output
                memory_gate = self.memory_gate(x)
                working_memory = memory_gate * working_memory + (1 - memory_gate) * x.mean(dim=1, keepdim=True)
            
            jump_prob = policy[:, 1] if i < self.layer - 1 else torch.zeros_like(policy[:, 1])
            should_jump = (torch.rand_like(jump_prob) < jump_prob).any()
            
            if should_jump:
                jump_length = torch.multinomial(policy, 1)[:, 0].max().item() + 1
                i_next = min(i + jump_length, self.layer - 1)
                skip_weight = self.jump_weights[min(jump_length-1, 2)]
                x = x + skip_weight * original_x + (1-skip_weight) * working_memory
                i = i_next
                jump_history.append(i)
            else:
                i += 1
        
        mlp_importance = self.mlp_gate(x)
        mlp_output = self.mlp(self.mlp_ln(x))
        x = x + mlp_importance * mlp_output
        return x, {'jumps': jump_history}
