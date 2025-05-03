import os
import re
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_citations_from_bbl(bbl_content):
    """
    Extract citations from .bbl file content.
    Returns a list of cited paper titles.
    """
    # Pattern to match bibitem entries
    pattern = r'\\bibitem\[(.*?)\]\[(.*?)\]\[(.*?)\]{(.*?)}'
    citations = []
    
    for match in re.finditer(pattern, bbl_content, re.DOTALL):
        # Extract the citation title from the bibitem content
        citation_content = match.group(0)
        # Try to extract the paper title from the content
        title_match = re.search(r'\\newblock\s*([^{}]*?)(?:\.|\\newblock|$)', citation_content)
        if title_match:
            title = title_match.group(1).strip()
            # Clean up the title
            title = re.sub(r'\\emph\{([^}]*)\}', r'\1', title)  # Remove \emph{}
            title = re.sub(r'\\newblock', '', title)  # Remove \newblock
            title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
            if title:
                print(title)
                citations.append(title)
    
    return citations

def build_citation_graph(dataset_path):
    """
    Build a citation graph from the dataset.
    Returns a networkx DiGraph and a title to paper_id mapping.
    """
    G = nx.DiGraph()
    paper_id_counter = 0
    title_to_id = {}
    id_to_title = {}
    id_to_folder = {}
    
    # First pass: create nodes for all papers
    for paper_folder in os.listdir(dataset_path):
        folder_path = os.path.join(dataset_path, paper_folder)
        if os.path.isdir(folder_path):
            title_path = os.path.join(folder_path, "title.txt")
            if os.path.exists(title_path):
                with open(title_path, 'r', encoding='utf-8') as f:
                    title = f.read().strip()
                    if title:
                        paper_id = paper_id_counter
                        title_to_id[title] = paper_id
                        id_to_title[paper_id] = title
                        id_to_folder[paper_id] = paper_folder
                        G.add_node(paper_id, title=title, folder=paper_folder)
                        paper_id_counter += 1
    
    # Second pass: add edges based on citations
    for paper_id, data in G.nodes(data=True):
        paper_folder = data['folder']
        folder_path = os.path.join(dataset_path, paper_folder)
        
        # Look for .bbl files in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith('.bbl'):
                bbl_path = os.path.join(folder_path, filename)
                with open(bbl_path, 'r', encoding='utf-8') as f:
                    bbl_content = f.read()
                    cited_titles = extract_citations_from_bbl(bbl_content)
                    
                    for cited_title in cited_titles:
                        if cited_title in title_to_id:
                            cited_id = title_to_id[cited_title]
                            G.add_edge(paper_id, cited_id)
    
    return G, title_to_id, id_to_title, id_to_folder

def analyze_graph(G):
    """
    Analyze the citation graph and print statistics.
    """
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    
    # Calculate in-degree and out-degree statistics
    in_degrees = [d for n, d in G.in_degree()]
    out_degrees = [d for n, d in G.out_degree()]
    
    print(f"Average in-degree: {sum(in_degrees)/len(in_degrees):.2f}")
    print(f"Average out-degree: {sum(out_degrees)/len(out_degrees):.2f}")
    
    # Number of isolated nodes (no citations in or out)
    isolated = [n for n in G.nodes() if G.in_degree(n) == 0 and G.out_degree(n) == 0]
    print(f"Number of isolated nodes: {len(isolated)}")
    
    # Plot degree distribution
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(in_degrees, bins=50, log=True)
    plt.title('In-degree Distribution (log scale)')
    plt.xlabel('In-degree')
    plt.ylabel('Count')
    
    plt.subplot(1, 2, 2)
    plt.hist(out_degrees, bins=50, log=True)
    plt.title('Out-degree Distribution (log scale)')
    plt.xlabel('Out-degree')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('degree_distribution.png')
    plt.close()
    
    # Calculate weakly connected components (for diameter estimation)
    if nx.is_weakly_connected(G):
        print("Graph is weakly connected")
        try:
            diameter = nx.diameter(G.to_undirected())
            print(f"Diameter of the graph: {diameter}")
        except:
            print("Graph is too large to compute diameter directly")
    else:
        print("Graph is not weakly connected")
        components = list(nx.weakly_connected_components(G))
        print(f"Number of weakly connected components: {len(components)}")
        largest_component = max(components, key=len)
        print(f"Size of largest component: {len(largest_component)}")
        
        # Compute diameter of largest component
        if len(largest_component) > 1:
            subgraph = G.subgraph(largest_component).to_undirected()
            try:
                diameter = nx.diameter(subgraph)
                print(f"Diameter of largest component: {diameter}")
            except:
                print("Largest component is too large to compute diameter directly")

if __name__ == "__main__":
    dataset_path = "./dataset_papers"  # Path to the extracted dataset
    G, title_to_id, id_to_title, id_to_folder = build_citation_graph(dataset_path)
    
    # Save the graph for later use
    nx.write_gpickle(G, "citation_graph.gpickle")
    
    # Analyze the graph
    analyze_graph(G)