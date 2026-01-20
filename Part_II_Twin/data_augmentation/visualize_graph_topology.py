#!/usr/bin/env python3
"""
Visualize FET heterogeneous graph topology with color-coded nodes and edges.
"""
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Patch
import pickle
from pathlib import Path
import argparse

# Node type categories and colors
NODE_CATEGORIES = {
    "Material Components": {
        "nodes": ["channel", "gate_top", "gate_bottom", "dielectric_top", "dielectric_bottom",
                  "floating_gate", "source", "drain", "substrate"],
        "color": "#FF6B6B",  # Red
    },
    "Sensing Components": {
        "nodes": ["surface_functionalization", "probe_material", "detect_target", "test_medium", "electrolyte"],
        "color": "#4ECDC4",  # Teal
    },
    "Process/Condition": {
        "nodes": ["process_annealing", "condition"],
        "color": "#FFE66D",  # Yellow
    },
}

# Edge type colors and styles
EDGE_STYLES = {
    "electrical": {"color": "#1A535C", "style": "solid", "width": 3, "label": "Electrical"},
    "capacitive": {"color": "#FF6B35", "style": "dashed", "width": 2.5, "label": "Capacitive"},
    "chemical": {"color": "#00A896", "style": "dotted", "width": 2, "label": "Chemical"},
    "process": {"color": "#9B5DE5", "style": "solid", "width": 2, "label": "Process"},
    "condition": {"color": "#F15BB5", "style": "solid", "width": 2, "label": "Condition"},
    "environment": {"color": "#FEE440", "style": "dashdot", "width": 1.5, "label": "Environment"},
}


def get_node_color(node_name):
    """Get color for a node based on its category."""
    for category, info in NODE_CATEGORIES.items():
        if node_name in info["nodes"]:
            return info["color"]
    return "#CCCCCC"  # Default gray


def visualize_graph_from_pickle(pkl_path, design_type="standard", output_file=None):
    """
    Visualize a graph from pickle file.

    Args:
        pkl_path: Path to the .pkl file
        design_type: Which design type to visualize (standard/remote/floating_gate/dual_gate)
        output_file: Optional output file path for saving the figure
    """
    # Load one sample graph
    with open(pkl_path, "rb") as f:
        graphs = pickle.load(f)

    if not graphs:
        print("No graphs found in pickle file")
        return

    sample_graph = graphs[0]

    # Build NetworkX graph
    G = nx.DiGraph()

    # Add nodes
    if isinstance(sample_graph, dict):
        nodes = sample_graph["nodes"].keys()
        edges = sample_graph["edges"]
    else:
        # PyTorch HeteroData
        nodes = sample_graph.node_types
        edges = {k: sample_graph[k].edge_index.numpy() for k in sample_graph.edge_types}

    for node in nodes:
        G.add_node(node, color=get_node_color(node))

    # Add edges with attributes
    edge_list_by_type = {}
    for (src, rel, dst), edge_index in edges.items():
        if edge_index.shape[1] == 0:
            continue
        if rel not in edge_list_by_type:
            edge_list_by_type[rel] = []
        # Add edge (ignore self-loops for clarity)
        if src in G.nodes and dst in G.nodes:
            G.add_edge(src, dst, rel=rel)
            edge_list_by_type[rel].append((src, dst))

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 12))

    # Draw nodes by category
    for category, info in NODE_CATEGORIES.items():
        node_list = [n for n in info["nodes"] if n in G.nodes]
        if node_list:
            nx.draw_networkx_nodes(
                G, pos, nodelist=node_list,
                node_color=info["color"],
                node_size=2000,
                alpha=0.9,
                ax=ax
            )

    # Draw edges by type
    for edge_type, edge_list in edge_list_by_type.items():
        if edge_type in EDGE_STYLES:
            style = EDGE_STYLES[edge_type]
            nx.draw_networkx_edges(
                G, pos, edgelist=edge_list,
                edge_color=style["color"],
                style=style["style"],
                width=style["width"],
                alpha=0.7,
                arrows=True,
                arrowsize=15,
                connectionstyle="arc3,rad=0.1",
                ax=ax
            )

    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=9, font_weight="bold", ax=ax)

    # Create legend
    legend_elements = []

    # Node categories
    for category, info in NODE_CATEGORIES.items():
        legend_elements.append(Patch(facecolor=info["color"], label=category, alpha=0.9))

    legend_elements.append(Patch(facecolor="white", label=""))  # Spacer

    # Edge types
    for edge_type, style in EDGE_STYLES.items():
        if edge_type in edge_list_by_type:
            legend_elements.append(
                Patch(
                    facecolor=style["color"],
                    label=f"{style['label']} edge",
                    alpha=0.7
                )
            )

    ax.legend(
        handles=legend_elements,
        loc="upper left",
        fontsize=11,
        framealpha=0.95,
        title="Node & Edge Types",
        title_fontsize=12
    )

    ax.set_title(
        f"FET Sensor Heterogeneous Graph Topology\n(Design: {design_type}, V2 topology with expert edges)",
        fontsize=16,
        fontweight="bold",
        pad=20
    )
    ax.axis("off")

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"Saved to {output_file}")
    else:
        plt.show()


def visualize_all_designs(output_dir="graph_topology_visualizations"):
    """Generate topology visualizations for all design types."""
    Path(output_dir).mkdir(exist_ok=True)

    # Find available graph files (look in current directory)
    graph_dir = Path(__file__).resolve().parent
    pkl_files = list(graph_dir.glob("graphs_*.pkl"))

    if not pkl_files:
        print("No graph pickle files found in current directory")
        return

    # Use the first available file
    pkl_file = pkl_files[0]
    print(f"Using graph file: {pkl_file}")

    # Visualize standard design
    output_file = Path(output_dir) / "graph_topology_standard.png"
    visualize_graph_from_pickle(pkl_file, design_type="standard", output_file=output_file)

    print(f"\nTopology visualization saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Visualize FET graph topology")
    parser.add_argument(
        "--graph-file",
        type=str,
        help="Path to graph pickle file (default: auto-detect from experiment-aug/)"
    )
    parser.add_argument(
        "--design",
        type=str,
        default="standard",
        choices=["standard", "remote", "floating_gate", "dual_gate"],
        help="Design type to visualize"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (if not specified, will display interactively)"
    )

    args = parser.parse_args()

    if args.graph_file:
        pkl_path = Path(args.graph_file)
    else:
        # Auto-detect (look in script directory)
        graph_dir = Path(__file__).resolve().parent
        pkl_files = list(graph_dir.glob("graphs_*.pkl"))
        if not pkl_files:
            print("ERROR: No graph files found in current directory")
            print("Please run build_graph_dataset_augmented.py first or specify --graph-file")
            return
        pkl_path = pkl_files[0]
        print(f"Auto-detected graph file: {pkl_path}")

    if not pkl_path.exists():
        print(f"ERROR: Graph file not found: {pkl_path}")
        return

    visualize_graph_from_pickle(pkl_path, design_type=args.design, output_file=args.output)


if __name__ == "__main__":
    main()
