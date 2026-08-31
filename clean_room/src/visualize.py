import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import os

def plot_mule_chain(chain_id='auto'):
    print("Generating Mule Flow Diagram...")
    df_txn = pd.read_csv('data/transactions.csv')
    
    if chain_id == 'auto':
        # Find a good complex chain (e.g., Split/Reconverge or Structuring)
        complex_chains = df_txn[df_txn['chain_id'].str.contains('SPLIT', na=False)]['chain_id'].unique()
        chain_id = complex_chains[0] if len(complex_chains) > 0 else df_txn['chain_id'].iloc[0]
        
    chain_txns = df_txn[df_txn['chain_id'] == chain_id].copy()
    
    G = nx.DiGraph()
    for _, row in chain_txns.iterrows():
        G.add_edge(row['source_account'], row['target_account'], amount=row['amount'])
        
    # Assign layers for a left-to-right flow chart
    for layer, nodes in enumerate(nx.topological_generations(G)):
        for node in nodes:
            G.nodes[node]['layer'] = layer
            
    pos = nx.multipartite_layout(G, subset_key='layer', align='horizontal')
    
    plt.figure(figsize=(14, 7))
    
    # Colors: Victim=Green, Cashout=DarkRed, Mule=Salmon
    node_colors = []
    for node in G.nodes():
        if "VICTIM" in str(node): node_colors.append('#90EE90') # Light green
        elif not list(G.out_edges(node)): node_colors.append('#8B0000') # Dark red
        else: node_colors.append('#FA8072') # Salmon
        
    # Node labels: shorten IDs for clean look
    labels = {n: n.split('_')[-1] if 'EXT' not in n else "VICTIM" for n in G.nodes()}
    
    nx.draw(G, pos, labels=labels, node_color=node_colors, node_size=3000, 
            font_size=10, font_weight='bold', edge_color='gray', 
            arrows=True, arrowsize=20)
            
    edge_labels = {(u, v): f"₹{d['amount']:,.0f}" for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9, font_color='black')
    
    plt.title(f"UPI Mule Flow Graph\nChain ID: {chain_id}", fontsize=14, fontweight='bold', pad=20)
    
    # Legend
    import matplotlib.patches as mpatches
    leg1 = mpatches.Patch(color='#90EE90', label='Victim (Source)')
    leg2 = mpatches.Patch(color='#FA8072', label='Intermediate Mule')
    leg3 = mpatches.Patch(color='#8B0000', label='Cashout Node')
    plt.legend(handles=[leg1, leg2, leg3], loc='lower right')
    
    os.makedirs('outputs', exist_ok=True)
    plt.tight_layout()
    plt.savefig('outputs/mule_flow.png', dpi=300, bbox_inches='tight')
    print("Saved mule_flow.png")
    
def plot_metrics_summary():
    print("Generating Metrics Summary...")
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis('tight')
    ax.axis('off')
    
    data = [
        ["Phase 3 (MVP Baseline)", "54.5%", "85.7%", "0.667", "15"],
        ["v1 (Invalidated - Tuned on Test)", "85.2%", "100%", "0.920", "4"],
        ["v2 (Final - Honest 60/20/20)", "74.1%", "95.2%", "0.833", "7"]
    ]
    columns = ["Model", "Precision", "Recall", "F1 Score", "False Positives"]
    
    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2.5)
    
    # Header color
    for i in range(len(columns)):
        table[(0, i)].set_facecolor("#40466e")
        table[(0, i)].get_text().set_color("white")
        table[(0, i)].set_text_props(weight='bold')
        
    plt.title("Track 2 Evaluation Metrics: MVP vs Multi-Hop Tracer", fontsize=14, fontweight='bold', pad=20)
    plt.savefig('outputs/metrics_summary.png', dpi=300, bbox_inches='tight')
    print("Saved metrics_summary.png")

if __name__ == "__main__":
    plot_mule_chain()
    plot_metrics_summary()
