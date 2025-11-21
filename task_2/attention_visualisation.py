"""
Attention Mechanism Visualization for Transformers
Cybersecurity Application: Command Sequence Analysis

This script generates visualizations of the self-attention mechanism
used in Transformer networks for security applications.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def softmax(x):
    """Compute softmax values for each row of x."""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V):
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: Query matrix (seq_len, d_k)
        K: Key matrix (seq_len, d_k)
        V: Value matrix (seq_len, d_v)
    
    Returns:
        output: Attention output
        attention_weights: Attention weight matrix
    """
    d_k = K.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attention_weights = softmax(scores)
    output = np.matmul(attention_weights, V)
    return output, attention_weights

def visualize_attention_mechanism():
    """
    Visualize the complete attention mechanism with a cybersecurity example.
    """
    # Example: Analyzing a potentially malicious command sequence
    tokens = ["cmd.exe", "/c", "powershell", "-enc", "BASE64STR"]
    seq_len = len(tokens)
    d_model = 8  # Embedding dimension
    
    # Simulated embeddings (in practice, these come from learned embeddings)
    np.random.seed(42)
    embeddings = np.random.randn(seq_len, d_model)
    
    # Weight matrices for Q, K, V projections
    W_q = np.random.randn(d_model, d_model) * 0.1
    W_k = np.random.randn(d_model, d_model) * 0.1
    W_v = np.random.randn(d_model, d_model) * 0.1
    
    # Compute Q, K, V
    Q = np.matmul(embeddings, W_q)
    K = np.matmul(embeddings, W_k)
    V = np.matmul(embeddings, W_v)
    
    # Compute attention
    output, attention_weights = scaled_dot_product_attention(Q, K, V)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Self-Attention Mechanism Visualization\nCybersecurity: Command Analysis', 
                 fontsize=14, fontweight='bold')
    
    # 1. Raw Attention Scores (before softmax)
    ax1 = axes[0, 0]
    raw_scores = np.matmul(Q, K.T) / np.sqrt(d_model)
    sns.heatmap(raw_scores, annot=True, fmt='.2f', cmap='YlOrRd',
                xticklabels=tokens, yticklabels=tokens, ax=ax1)
    ax1.set_title('Raw Attention Scores (Q·K^T / √d_k)')
    ax1.set_xlabel('Key Tokens')
    ax1.set_ylabel('Query Tokens')
    
    # 2. Attention Weights (after softmax)
    ax2 = axes[0, 1]
    sns.heatmap(attention_weights, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=tokens, yticklabels=tokens, ax=ax2)
    ax2.set_title('Attention Weights (Softmax Applied)')
    ax2.set_xlabel('Key Tokens')
    ax2.set_ylabel('Query Tokens')
    
    # 3. Attention Pattern Analysis
    ax3 = axes[1, 0]
    # Highlight which tokens each position attends to most
    max_attention_idx = np.argmax(attention_weights, axis=1)
    attention_pattern = np.zeros_like(attention_weights)
    for i, j in enumerate(max_attention_idx):
        attention_pattern[i, j] = 1
    
    sns.heatmap(attention_pattern, cmap='Greens', 
                xticklabels=tokens, yticklabels=tokens, ax=ax3,
                cbar_kws={'label': 'Max Attention'})
    ax3.set_title('Strongest Attention Connections')
    ax3.set_xlabel('Attended Token')
    ax3.set_ylabel('Query Token')
    
    # Add arrows showing attention flow
    for i in range(seq_len):
        j = max_attention_idx[i]
        ax3.annotate('', xy=(j + 0.5, i + 0.5), xytext=(i + 0.5, i + 0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # 4. Embedding Space Visualization
    ax4 = axes[1, 1]
    # Use PCA-like reduction for visualization
    from numpy.linalg import svd
    U, S, Vt = svd(embeddings)
    reduced = U[:, :2] * S[:2]
    
    colors = ['blue', 'gray', 'red', 'orange', 'purple']
    for i, (token, color) in enumerate(zip(tokens, colors)):
        ax4.scatter(reduced[i, 0], reduced[i, 1], c=color, s=200, 
                   label=token, edgecolors='black', linewidth=2)
        ax4.annotate(token, (reduced[i, 0] + 0.05, reduced[i, 1] + 0.05),
                    fontsize=10, fontweight='bold')
    
    # Draw attention connections
    for i in range(seq_len):
        for j in range(seq_len):
            if attention_weights[i, j] > 0.2:  # Threshold
                ax4.plot([reduced[i, 0], reduced[j, 0]], 
                        [reduced[i, 1], reduced[j, 1]],
                        'k-', alpha=attention_weights[i, j], linewidth=2)
    
    ax4.set_title('Token Embeddings with Attention Connections')
    ax4.set_xlabel('Dimension 1')
    ax4.set_ylabel('Dimension 2')
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return attention_weights

def visualize_multi_head_attention():
    """
    Visualize multi-head attention with different attention patterns.
    """
    tokens = ["GET", "/admin", "HTTP/1.1", "SELECT", "*", "FROM", "users"]
    seq_len = len(tokens)
    num_heads = 4
    
    np.random.seed(123)
    
    # Simulate different attention patterns for each head
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Multi-Head Attention: Different Heads Focus on Different Patterns\n'
                 'Example: SQL Injection Detection in HTTP Request', 
                 fontsize=12, fontweight='bold')
    
    head_names = [
        'Head 1: HTTP Structure',
        'Head 2: SQL Keywords', 
        'Head 3: Sequential Flow',
        'Head 4: Anomaly Detection'
    ]
    
    # Create different attention patterns
    patterns = []
    
    # Head 1: Focus on HTTP structure
    p1 = np.array([
        [0.4, 0.3, 0.2, 0.05, 0.02, 0.02, 0.01],
        [0.3, 0.4, 0.2, 0.05, 0.02, 0.02, 0.01],
        [0.2, 0.2, 0.5, 0.05, 0.02, 0.02, 0.01],
        [0.1, 0.1, 0.1, 0.3, 0.15, 0.15, 0.1],
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.1],
        [0.1, 0.1, 0.1, 0.15, 0.15, 0.3, 0.1],
        [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.3]
    ])
    patterns.append(p1)
    
    # Head 2: Focus on SQL keywords
    p2 = np.array([
        [0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1],
        [0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1],
        [0.1, 0.1, 0.1, 0.4, 0.15, 0.15, 0.0],
        [0.05, 0.05, 0.05, 0.25, 0.3, 0.15, 0.15],
        [0.05, 0.05, 0.05, 0.2, 0.15, 0.35, 0.15],
        [0.05, 0.1, 0.05, 0.15, 0.1, 0.2, 0.35]
    ])
    patterns.append(p2)
    
    # Head 3: Sequential flow
    p3 = np.eye(seq_len) * 0.4 + 0.1
    p3 = p3 / p3.sum(axis=1, keepdims=True)
    patterns.append(p3)
    
    # Head 4: Anomaly (SQL after HTTP)
    p4 = np.array([
        [0.3, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1],
        [0.1, 0.2, 0.1, 0.25, 0.1, 0.15, 0.1],
        [0.05, 0.05, 0.2, 0.35, 0.1, 0.15, 0.1],
        [0.05, 0.15, 0.2, 0.3, 0.1, 0.1, 0.1],
        [0.05, 0.1, 0.15, 0.25, 0.2, 0.15, 0.1],
        [0.05, 0.1, 0.1, 0.2, 0.15, 0.25, 0.15],
        [0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.25]
    ])
    patterns.append(p4)
    
    for idx, (ax, pattern, name) in enumerate(zip(axes.flat, patterns, head_names)):
        sns.heatmap(pattern, annot=True, fmt='.2f', cmap='Purples',
                    xticklabels=tokens, yticklabels=tokens, ax=ax)
        ax.set_title(name)
        ax.set_xlabel('Key')
        ax.set_ylabel('Query')
    
    plt.tight_layout()
    plt.savefig('multi_head_attention.png', dpi=150, bbox_inches='tight')
    plt.show()

def attention_flow_diagram():
    """
    Create a diagram showing the flow of attention computation.
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title('Self-Attention Computation Flow', fontsize=16, fontweight='bold')
    
    # Input embeddings
    ax.add_patch(plt.Rectangle((1, 8), 2, 1.5, fill=True, color='lightblue', ec='black'))
    ax.text(2, 8.75, 'Input\nEmbeddings\n(X)', ha='center', va='center', fontsize=10)
    
    # Weight matrices
    for i, (label, color) in enumerate([('W_Q', 'lightcoral'), 
                                         ('W_K', 'lightgreen'), 
                                         ('W_V', 'lightyellow')]):
        ax.add_patch(plt.Rectangle((4 + i*2, 8), 1.5, 1.5, fill=True, color=color, ec='black'))
        ax.text(4.75 + i*2, 8.75, label, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Q, K, V
    for i, (label, color) in enumerate([('Q', 'coral'), ('K', 'green'), ('V', 'yellow')]):
        ax.add_patch(plt.Rectangle((4 + i*2, 5.5), 1.5, 1.5, fill=True, color=color, ec='black'))
        ax.text(4.75 + i*2, 6.25, label, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Arrows from input to weight matrices
    for i in range(3):
        ax.annotate('', xy=(4.75 + i*2, 8), xytext=(2.5, 8),
                   arrowprops=dict(arrowstyle='->', color='black'))
    
    # Arrows from weight matrices to Q, K, V
    for i in range(3):
        ax.annotate('', xy=(4.75 + i*2, 7), xytext=(4.75 + i*2, 8),
                   arrowprops=dict(arrowstyle='->', color='black'))
    
    # QK^T operation
    ax.add_patch(plt.Rectangle((5, 3.5), 2, 1.2, fill=True, color='lavender', ec='black'))
    ax.text(6, 4.1, 'Q · K^T / √d_k', ha='center', va='center', fontsize=10)
    
    # Softmax
    ax.add_patch(plt.Rectangle((5, 2), 2, 1, fill=True, color='plum', ec='black'))
    ax.text(6, 2.5, 'Softmax', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Attention weights
    ax.add_patch(plt.Rectangle((5, 0.5), 2, 1, fill=True, color='orchid', ec='black'))
    ax.text(6, 1, 'Attention\nWeights', ha='center', va='center', fontsize=9)
    
    # Multiply with V
    ax.add_patch(plt.Rectangle((9, 2), 2.5, 1.5, fill=True, color='gold', ec='black'))
    ax.text(10.25, 2.75, 'Weights × V', ha='center', va='center', fontsize=10)
    
    # Output
    ax.add_patch(plt.Rectangle((12.5, 2), 2.5, 1.5, fill=True, color='lightsteelblue', ec='black'))
    ax.text(13.75, 2.75, 'Output\nEmbeddings', ha='center', va='center', fontsize=10)
    
    # Connecting arrows
    ax.annotate('', xy=(5, 4.1), xytext=(4.75, 5.5), arrowprops=dict(arrowstyle='->', color='coral'))
    ax.annotate('', xy=(7, 4.1), xytext=(6.75, 5.5), arrowprops=dict(arrowstyle='->', color='green'))
    ax.annotate('', xy=(6, 3.5), xytext=(6, 3.5), arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('', xy=(6, 2), xytext=(6, 3.5), arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('', xy=(6, 1.5), xytext=(6, 2), arrowprops=dict(arrowstyle='->', color='black'))
    ax.annotate('', xy=(9, 2.75), xytext=(7, 1), arrowprops=dict(arrowstyle='->', color='purple'))
    ax.annotate('', xy=(9, 2.75), xytext=(8.75, 5.5), arrowprops=dict(arrowstyle='->', color='olive'))
    ax.annotate('', xy=(12.5, 2.75), xytext=(11.5, 2.75), arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig('attention_flow.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    print("Generating Attention Mechanism Visualizations...")
    print("=" * 50)
    
    print("\n1. Basic Self-Attention Visualization")
    attention_weights = visualize_attention_mechanism()
    
    print("\n2. Multi-Head Attention Visualization")
    visualize_multi_head_attention()
    
    print("\n3. Attention Flow Diagram")
    attention_flow_diagram()
    
    print("\n" + "=" * 50)
    print("Visualizations saved:")
    print("- attention_visualization.png")
    print("- multi_head_attention.png")
    print("- attention_flow.png")
