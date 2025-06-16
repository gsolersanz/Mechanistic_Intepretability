import torch
import numpy as np
import matplotlib.pyplot as plt
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import torch.nn.functional as F

def analyze_model_predictions(clean_text, corrupted_text):
    # Load pretrained model
    model = GPT.from_pretrained('gpt2')
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    tokenizer = BPETokenizer()

    # Tokenization
    clean_tokens = tokenizer(clean_text).to(device)
    corrupted_tokens = tokenizer(corrupted_text).to(device)
    assert clean_tokens.shape == corrupted_tokens.shape, "Texts must have the same number of tokens"

    # Get token indices (add space before names as they're not at start of sequence)
    cats_index = tokenizer(' Paris')[0].item()
    tigers_index = tokenizer(' Milan')[0].item()
    
    print(f"Paris index: {cats_index}, Milan index: {tigers_index}")

    # First run: get clean activations
    with torch.no_grad():
        _, _ = model(clean_tokens, save_activations=True)
        clean_activations = model.stored_activations

        # Print probabilities for the clean run
        logits = model.last_token_logits
        probs = F.softmax(logits, dim=-1)
        top_probs, top_tokens = torch.topk(probs[0], k=20)
        print("\nTop 20 predictions for clean input:")
        for i, (prob, token) in enumerate(zip(top_probs, top_tokens)):
            token_str = tokenizer.decode(torch.tensor([token]))
            print(f"{i+1:2d} {token.item():5d} {token_str:15s} {prob.item():.4f}")

    # Second run: get baseline corrupted prediction
    with torch.no_grad():
        _, _ = model(corrupted_tokens)
        baseline_logits = model.last_token_logits
        baseline_diff = baseline_logits[0, tigers_index] - baseline_logits[0, cats_index]
        print(f"\nBaseline logit difference (Tigers - Cats): {baseline_diff.item():.4f}")

    # Initialize difference matrix and predictions container
    num_layers = len(model.transformer.h)
    num_positions = clean_tokens.shape[1]
    difference_matrix = torch.zeros((num_layers, num_positions))
    patched_predictions = {}

    # Perform patching for each layer and position
    for layer in range(num_layers):
        for position in range(num_positions):
            with torch.no_grad():
                # Run model with current patch
                _, _ = model(
                    corrupted_tokens,
                    patch_layer=layer,
                    patch_position=position,
                    clean_activations=clean_activations
                )
                patched_logits = model.last_token_logits
                
                # Calculate Milan - Paris difference in logits (not probabilities)
                current_diff = patched_logits[0, tigers_index] - patched_logits[0, cats_index]
                difference_matrix[layer, position] = current_diff.item()
                
                # Store predictions for this patch
                probs = F.softmax(patched_logits, dim=-1)
                top_probs, top_tokens = torch.topk(probs[0], k=5)  # Get top 5 predictions
                predictions = [(tokenizer.decode(torch.tensor([t])), p.item()) for t, p in zip(top_tokens, top_probs)]
                patched_predictions[(layer, position)] = predictions

                # Optional: Print large differences for debugging
                if abs(difference_matrix[layer, position]) > 1.0:
                    print(f"Large difference at layer {layer}, position {position}: {difference_matrix[layer, position]:.2f}")

    return difference_matrix.numpy(), [tokenizer.decode(torch.tensor([t])) for t in corrupted_tokens[0]], patched_predictions

def plot_difference_matrix(difference_matrix, tokens):
    import matplotlib.pyplot as plt
    import numpy as np

    plt.figure(figsize=(8, 8))  # Compact size for the plot

    # Symmetric log scale for better visualization
    abs_max = max(abs(difference_matrix.min()), abs(difference_matrix.max()))
    vmin, vmax = -abs_max, abs_max

    plt.imshow(difference_matrix,
               cmap='cividis',  # Enhanced color map
               aspect='auto',
               interpolation='nearest',
               vmin=vmin,
               vmax=vmax)

    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Logit Difference', fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Titles and labels
    plt.title("Impact of Activation Patching on Token Prediction", fontsize=14, pad=15)
    plt.xlabel("Token Position", fontsize=12)
    plt.ylabel("Layer", fontsize=12)

    # Configure axis labels
    plt.xticks(
        range(len(tokens)),
        tokens,
        rotation=45,
        ha='right',
        fontsize=10
    )
    plt.yticks(
        range(difference_matrix.shape[0]),
        [f"layer {i}" for i in range(difference_matrix.shape[0])],
        fontsize=10
    )

    plt.tight_layout()
    plt.show()

#clean_text = "If you say that cats and dogs falling from the sky, this means the weather is"
#clean_text = "When someone says that cats and dogs are falling from the sky, it means it's going to"
#corrupted_text = "When someone says that tigers and dogs are falling from the sky, it means it's going to"
#corrupted_text = "When someone says it's raining elephants and dogs, it means the weather is"
clean_text ='The Paris Eiffel Tower, a landmark in Paris, is situated in the country of'


corrupted_text ='The Milan Eiffel Tower, a landmark in Paris, is situated in the country of'


diff_matrix, tokens, patched_preds = analyze_model_predictions(clean_text, corrupted_text)
plot_difference_matrix(diff_matrix, tokens)

# Print patched predictions for debugging
for (layer, position), preds in patched_preds.items():
    print(f"Layer {layer}, Position {position} - Top Predictions: {preds}")
