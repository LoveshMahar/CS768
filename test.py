import os
import re
import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle


def try_decode(content_bytes, encodings=('utf-8', 'latin-1', 'cp1252', 'iso-8859-1')):
    """
    Try decoding bytes with multiple encodings.
    Returns decoded content or None if all attempts fail.
    """
    for encoding in encodings:
        try:
            return content_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return None

def read_file_with_fallback(filepath):
    """
    Read a file with encoding fallback.
    Returns content or None if decoding fails.
    """
    try:
        with open(filepath, 'rb') as f:
            content_bytes = f.read()
        return try_decode(content_bytes)
    except (IOError, UnicodeDecodeError):
        return None


def extract_citations_from_bbl(bbl_content):
    """
    Extract citations from .bbl file content.
    Returns a list of cited paper titles.
    """
    if not bbl_content:
        return []

    # Pattern to match complete bibitem entries including all content until next \bibitem or \end
    pattern = r'\\bibitem(?:\[[^\]]*\]){0,3}\{[^\}]*\}(.*?)(?=\\bibitem|\\end\{thebibliography\})'
    citations = []
    
    for match in re.finditer(pattern, bbl_content, re.DOTALL):
        entry_content = match.group(1).strip()
        
        # Extract title from the first \newblock (main title) or the first line if no \newblock
        title = None
        newblock_match = re.search(r'\\newblock\s*(.*?)(?=\\newblock|$)', entry_content, re.DOTALL)
        if newblock_match:
            title = newblock_match.group(1).strip()
        else:
            # Fallback: take first non-empty line
            lines = [line.strip() for line in entry_content.split('\n') if line.strip()]
            if lines:
                title = lines[0]
        
        if title:
            # Clean up the title
            # print(title)
            title = re.sub(r'\\emph\{([^}]*)\}', r'\1', title)  # Remove \emph{}
            title = re.sub(r'\\newblock', '', title)  # Remove \newblock
            title = re.sub(r'\$.*?\$', '', title)  # Remove math expressions
            title = re.sub(r'\\[^\s{}]*', '', title)  # Remove other LaTeX commands
            title = re.sub(r'\{|\}', '', title)  # Remove curly braces
            title = re.sub(r'\s+', ' ', title).strip()  # Normalize whitespace
            
            # Remove common trailing punctuation and metadata
            title = re.sub(r'\.$', '', title)
            title = re.sub(r'\(Tech\..*?\)$', '', title).strip()
            title = re.sub(r'\(.*?\)$', '', title).strip()
            
            if title:
                # print(title)
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
    problematic_files = []
    l = 0
    c = 0
    # First pass: create nodes for all papers
    for paper_folder in os.listdir(dataset_path):
        l = l + 1
        folder_path = os.path.join(dataset_path, paper_folder)
        if os.path.isdir(folder_path):
            title_path = os.path.join(folder_path, "title.txt")
            title_content = read_file_with_fallback(title_path)
            paper_id = -1
            if title_content:
                title = title_content.strip()
                if title:
                    if title in title_to_id:
                        paper_id = title_to_id[title]
                    else:
                        paper_id = paper_id_counter
                        title_to_id[title] = paper_id
                        id_to_title[paper_id] = title
                        id_to_folder[paper_id] = paper_folder
                        G.add_node(paper_id, title=title, folder=paper_folder)
                        paper_id_counter += 1
                    for filename in os.listdir(folder_path):
                        if filename.endswith('.bbl'):
                            bbl_path = os.path.join(folder_path, filename)
                            bbl_content = read_file_with_fallback(bbl_path)
                            
                            if bbl_content:
                                cited_titles = extract_citations_from_bbl(bbl_content)
                                
                                for cited_title in cited_titles:
                                    c += 1
                                    if cited_title in title_to_id:
                                        cited_id = title_to_id[cited_title]
                                        G.add_edge(paper_id, cited_id)
                                    else:
                                        paper_id1 = paper_id_counter
                                        title_to_id[cited_title] = paper_id1
                                        id_to_title[paper_id1] = cited_title
                                        G.add_node(paper_id1, title = cited_title, folder = paper_folder)
                                        G.add_edge(paper_id, paper_id1)
                                        paper_id_counter += 1
                            else:
                                problematic_files.append(bbl_path)

            else:
                problematic_files.append(title_path)
    # Second pass: add edges based on citations
    # print("number of titles: %d", c)
    # for paper_id, data in G.nodes(data=True):
    #     paper_folder = data['folder']
    #     folder_path = os.path.join(dataset_path, paper_folder)
        
    #     # Look for .bbl files in the folder
    #     for filename in os.listdir(folder_path):
    #         if filename.endswith('.bbl'):
    #             bbl_path = os.path.join(folder_path, filename)
    #             bbl_content = read_file_with_fallback(bbl_path)
                
    #             if bbl_content:
    #                 cited_titles = extract_citations_from_bbl(bbl_content)
                    
    #                 for cited_title in cited_titles:
    #                     if cited_title in title_to_id:
    #                         cited_id = title_to_id[cited_title]
    #                         G.add_edge(paper_id, cited_id)
    #                     else:
    #                         paper_id = paper_id_counter
    #                         title_to_id[title] = paper_id
    #                         id_to_title[paper_id] = title
    #                         G.add_node(paper_id, title = cited_title)
    #             else:
    #                 problematic_files.append(bbl_path)
    
    # Write problematic files to a log
    with open('problematic_files.log', 'w') as f:
        f.write("Files that couldn't be decoded:\n")
        f.write("\n".join(problematic_files))
    
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
    
    # Save the graph using pickle (recommended for NetworkX 3.0+)
    with open("citation_graph.gpickle", "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Analyze the graph
    analyze_graph(G)