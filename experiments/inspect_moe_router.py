"""
inspect_moe_router.py - A minimal, complete demonstration of:
1. The Gateway / Router architecture (just a single linear matrix!).
2. How gradients flow backward from the loss to train the Router to pick the best expert.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Fix random seed for clean demonstration
torch.manual_seed(42)

class ToyExpert(nn.Module):
    """A standard 2-layer MLP expert."""
    def __init__(self, d_model=16, d_ff=32):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class ToyMoELayer(nn.Module):
    """
    A Mixture of Experts layer with:
    - 4 Experts
    - 1 Gateway / Router (a single linear layer!)
    - Top-2 Routing
    """
    def __init__(self, d_model=16, num_experts=4, top_k=2):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # 1. THE GATEWAY ARCHITECTURE: Literally just ONE linear projection!
        # Shape: [d_model, num_experts] (e.g. 16 -> 4)
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # 2. THE EXPERTS
        self.experts = nn.ModuleList([ToyExpert(d_model) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: [batch_size, d_model]
        batch_size = x.shape[0]

        # Step 1: The Gateway calculates a raw score for each of the 4 experts
        # router_logits shape: [batch_size, 4]
        router_logits = self.router(x)

        # Step 2: Convert logits to probabilities with Softmax
        routing_weights = F.softmax(router_logits, dim=-1)

        # Step 3: Pick the Top-2 experts
        topk_weights, topk_indices = torch.topk(routing_weights, self.top_k, dim=-1)

        # Normalize the chosen weights so they sum to 1.0
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Step 4: Compute expert outputs and combine them weighted by topk_weights
        final_output = torch.zeros_like(x)
        for i in range(self.top_k):
            expert_idx = topk_indices[:, i]  # which expert is chosen
            weight = topk_weights[:, i].unsqueeze(-1)  # weight for that expert

            for b in range(batch_size):
                exp = self.experts[expert_idx[b]]
                expert_out = exp(x[b:b+1])
                final_output[b:b+1] += weight[b:b+1] * expert_out

        return final_output, router_logits, topk_indices, topk_weights


def main():
    print("=" * 70)
    print("DEMO: HOW THE MOE GATEWAY WORKS AND HOW IT GETS TRAINED")
    print("=" * 70)

    d_model = 8
    num_experts = 4
    moe = ToyMoELayer(d_model=d_model, num_experts=num_experts, top_k=2)

    # 1. Look at the Router Architecture
    print("\n1. THE GATEWAY ARCHITECTURE:")
    print(f"   Router Weight Matrix Shape: {moe.router.weight.shape}")
    print("   Notice: It is literally just a single matrix of numbers!")
    print(f"   Row count = {num_experts} (experts), Col count = {d_model} (token features)\n")

    # 2. Simulate 2 distinct types of tokens:
    # Token A: "Code token" (features point in one direction)
    # Token B: "Math token" (features point in another direction)
    token_code = torch.randn(1, d_model)
    token_math = torch.randn(1, d_model)

    print("2. BEFORE TRAINING (Random Weights):")
    with torch.no_grad():
        _, _, chosen_code, weights_code = moe(token_code)
        _, _, chosen_math, weights_math = moe(token_math)
    print(f"   Code Token was sent to Experts: {chosen_code[0].tolist()} with weights {weights_code[0].round(decimals=3).tolist()}")
    print(f"   Math Token was sent to Experts: {chosen_math[0].tolist()} with weights {weights_math[0].round(decimals=3).tolist()}")

    # 3. TRAIN THE GATEWAY!
    # Let's say: Target task requires Expert 2 to handle Code, and Expert 0 to handle Math!
    optimizer = torch.optim.Adam(moe.parameters(), lr=0.05)

    print("\n3. TRAINING THE ROUTER VIA BACKPROPAGATION (5 Steps)...")
    # We train the layer so that target outputs match
    target_code = token_code * 2.0
    target_math = token_math * -1.5

    for step in range(1, 6):
        optimizer.zero_grad()

        out_code, _, idx_code, _ = moe(token_code)
        out_math, _, idx_math, _ = moe(token_math)

        loss = F.mse_loss(out_code, target_code) + F.mse_loss(out_math, target_math)
        loss.backward()

        # Check router gradient
        router_grad_norm = moe.router.weight.grad.norm().item()
        optimizer.step()

        print(f"   Step {step}: Loss = {loss.item():.4f} | Router Gradient Norm = {router_grad_norm:.4f}")

    print("\n4. AFTER TRAINING (The Router has learned specialization!):")
    with torch.no_grad():
        _, _, chosen_code_after, weights_code_after = moe(token_code)
        _, _, chosen_math_after, weights_math_after = moe(token_math)

    print(f"   Code Token is now routed to: {chosen_code_after[0].tolist()} with weights {weights_code_after[0].round(decimals=3).tolist()}")
    print(f"   Math Token is now routed to: {chosen_math_after[0].tolist()} with weights {weights_math_after[0].round(decimals=3).tolist()}")

    print("\n" + "=" * 70)
    print("TAKEAWAY: The gateway learns because the final output Y = w1*E1 + w2*E2.")
    print("Calculus flows gradients directly from the error into the router's weights!")
    print("=" * 70)

if __name__ == "__main__":
    main()
