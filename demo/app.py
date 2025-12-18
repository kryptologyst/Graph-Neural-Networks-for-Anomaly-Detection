"""Interactive demo for anomaly detection using Streamlit."""

import streamlit as st
import torch
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pyvis.network import Network

from src.utils.device import get_device, set_seed
from src.data.dataset import load_dataset, get_data_info
from src.models.anomaly_models import create_model
from src.eval.metrics import compute_anomaly_scores


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="GNN Anomaly Detection Demo",
        page_icon="🔍",
        layout="wide",
    )
    
    st.title("🔍 Graph Neural Networks for Anomaly Detection")
    st.markdown("""
    This demo showcases anomaly detection in graphs using various GNN architectures.
    Explore different models, datasets, and visualize anomaly detection results.
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Dataset selection
    dataset_options = {
        "Synthetic (1000 nodes, 10% anomalies)": "synthetic_1000_barabasi_albert_0.1",
        "Synthetic (500 nodes, 5% anomalies)": "synthetic_500_barabasi_albert_0.05",
        "Synthetic (2000 nodes, 15% anomalies)": "synthetic_2000_barabasi_albert_0.15",
        "Cora Citation Network": "cora",
        "Karate Club": "karate",
    }
    
    selected_dataset = st.sidebar.selectbox(
        "Select Dataset",
        list(dataset_options.keys()),
        index=0,
    )
    dataset_name = dataset_options[selected_dataset]
    
    # Model selection
    model_options = {
        "Graph Autoencoder (GAE)": "gae",
        "Variational Graph Autoencoder (VGAE)": "vgae",
        "Graph Attention Network (GAT)": "gat_gae",
        "GraphSAGE": "sage_gae",
        "DOMINANT": "dominant",
    }
    
    selected_model = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0,
    )
    model_name = model_options[selected_model]
    
    # Load dataset and model
    if st.sidebar.button("Load Dataset & Model"):
        with st.spinner("Loading dataset and model..."):
            try:
                # Set seed for reproducibility
                set_seed(42)
                
                # Load dataset
                data, dataset_type = load_dataset(
                    dataset_name=dataset_name,
                    root="./data",
                )
                
                # Create model
                device = get_device()
                model = create_model(
                    model_name=model_name,
                    in_channels=data.num_node_features,
                    hidden_channels=64,
                    out_channels=32,
                ).to(device)
                
                # Store in session state
                st.session_state.data = data
                st.session_state.model = model
                st.session_state.device = device
                st.session_state.dataset_name = dataset_name
                st.session_state.model_name = model_name
                
                st.success("Dataset and model loaded successfully!")
                
            except Exception as e:
                st.error(f"Error loading dataset/model: {str(e)}")
    
    # Main content
    if 'data' in st.session_state and 'model' in st.session_state:
        data = st.session_state.data
        model = st.session_state.model
        device = st.session_state.device
        
        # Dataset information
        st.header("📊 Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", data.num_nodes)
        with col2:
            st.metric("Edges", data.num_edges)
        with col3:
            st.metric("Features", data.num_node_features)
        with col4:
            if hasattr(data, 'anomaly_labels'):
                num_anomalies = data.anomaly_labels.sum().item()
                st.metric("Anomalies", f"{num_anomalies} ({num_anomalies/data.num_nodes*100:.1f}%)")
            else:
                st.metric("Anomalies", "N/A")
        
        # Model information
        st.header("🤖 Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Model", st.session_state.model_name.upper())
        
        with col2:
            num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            st.metric("Parameters", f"{num_params:,}")
        
        # Anomaly detection
        st.header("🔍 Anomaly Detection")
        
        if st.button("Detect Anomalies"):
            with st.spinner("Computing anomaly scores..."):
                try:
                    # Prepare edge data
                    if hasattr(data, 'train_pos_edge_index'):
                        edge_index = data.train_pos_edge_index
                    else:
                        edge_index = data.edge_index
                    
                    # Compute anomaly scores
                    scores = compute_anomaly_scores(
                        model=model,
                        data=data.x,
                        edge_index=edge_index,
                        method="reconstruction",
                    )
                    
                    # Store scores
                    st.session_state.scores = scores
                    
                    st.success("Anomaly detection completed!")
                    
                except Exception as e:
                    st.error(f"Error during anomaly detection: {str(e)}")
        
        # Results visualization
        if 'scores' in st.session_state:
            scores = st.session_state.scores
            
            st.header("📈 Results Visualization")
            
            # Score distribution
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Anomaly Score Distribution")
                
                # Create histogram
                fig = go.Figure()
                fig.add_trace(go.Histogram(
                    x=scores.cpu().numpy(),
                    nbinsx=50,
                    name="All Nodes",
                    opacity=0.7,
                ))
                
                if hasattr(data, 'anomaly_labels'):
                    anomaly_scores = scores[data.anomaly_labels].cpu().numpy()
                    normal_scores = scores[~data.anomaly_labels].cpu().numpy()
                    
                    fig.add_trace(go.Histogram(
                        x=normal_scores,
                        nbinsx=50,
                        name="Normal Nodes",
                        opacity=0.7,
                    ))
                    
                    fig.add_trace(go.Histogram(
                        x=anomaly_scores,
                        nbinsx=50,
                        name="Anomalous Nodes",
                        opacity=0.7,
                    ))
                
                fig.update_layout(
                    title="Distribution of Anomaly Scores",
                    xaxis_title="Anomaly Score",
                    yaxis_title="Count",
                    barmode="overlay",
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("Top Anomalous Nodes")
                
                # Get top anomalous nodes
                top_k = st.slider("Number of top anomalies to show", 5, 50, 10)
                top_indices = scores.topk(top_k).indices
                top_scores = scores[top_indices]
                
                # Create dataframe
                df = pd.DataFrame({
                    'Node ID': top_indices.cpu().numpy(),
                    'Anomaly Score': top_scores.cpu().numpy(),
                })
                
                # Add ground truth if available
                if hasattr(data, 'anomaly_labels'):
                    df['Is Anomaly'] = data.anomaly_labels[top_indices].cpu().numpy()
                    df['Is Anomaly'] = df['Is Anomaly'].map({True: 'Yes', False: 'No'})
                
                st.dataframe(df, use_container_width=True)
            
            # Graph visualization
            st.subheader("Graph Visualization")
            
            # Create network visualization
            if st.button("Generate Graph Visualization"):
                with st.spinner("Creating graph visualization..."):
                    try:
                        # Convert to NetworkX
                        G = nx.from_edgelist(data.edge_index.t().cpu().numpy())
                        
                        # Add node attributes
                        for i, score in enumerate(scores.cpu().numpy()):
                            if i in G.nodes:
                                G.nodes[i]['score'] = score
                                if hasattr(data, 'anomaly_labels'):
                                    G.nodes[i]['is_anomaly'] = data.anomaly_labels[i].item()
                        
                        # Create PyVis network
                        net = Network(
                            height="500px",
                            width="100%",
                            bgcolor="#222222",
                            font_color="white",
                        )
                        
                        # Add nodes
                        for node in G.nodes():
                            score = G.nodes[node]['score']
                            is_anomaly = G.nodes[node].get('is_anomaly', False)
                            
                            # Color nodes based on anomaly status
                            if is_anomaly:
                                color = "red"
                                title = f"Node {node}\nAnomaly Score: {score:.4f}\nStatus: Anomalous"
                            else:
                                color = "blue"
                                title = f"Node {node}\nAnomaly Score: {score:.4f}\nStatus: Normal"
                            
                            net.add_node(
                                node,
                                label=str(node),
                                color=color,
                                title=title,
                                size=min(max(score * 10, 5), 30),
                            )
                        
                        # Add edges
                        for edge in G.edges():
                            net.add_edge(edge[0], edge[1])
                        
                        # Save and display
                        net.save_graph("temp_graph.html")
                        
                        with open("temp_graph.html", "r") as f:
                            html_content = f.read()
                        
                        st.components.v1.html(html_content, height=600)
                        
                    except Exception as e:
                        st.error(f"Error creating graph visualization: {str(e)}")
            
            # Performance metrics
            if hasattr(data, 'anomaly_labels'):
                st.subheader("Performance Metrics")
                
                from sklearn.metrics import roc_auc_score, average_precision_score
                
                auroc = roc_auc_score(data.anomaly_labels.cpu().numpy(), scores.cpu().numpy())
                auprc = average_precision_score(data.anomaly_labels.cpu().numpy(), scores.cpu().numpy())
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("AUROC", f"{auroc:.4f}")
                
                with col2:
                    st.metric("AUPRC", f"{auprc:.4f}")
    
    else:
        st.info("Please load a dataset and model using the sidebar controls.")


if __name__ == "__main__":
    main()
