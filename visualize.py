import argparse
import networkx as nx
import matplotlib.pyplot as plt
import os
from IPython import embed
from tqdm import tqdm


def parse_graph(line: str):
    graph_edges_str = line.split(')(')
    graph_edges_str[0] = graph_edges_str[0][1:]
    graph_edges_str[-1] = graph_edges_str[-1][:-1]
    
    graph = nx.Graph()
    
    for edge_str in graph_edges_str:
        edge = edge_str.split(';')
        edge = [part.strip() for part in edge]
        graph.add_edge(edge[0], edge[2], relation=edge[1])
        
    return graph
        
    
def visualize(correct_graph: nx.Graph, predicted_graph: nx.Graph, sentence: str,  path: str):
    
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))

    pos = nx.nx_agraph.graphviz_layout(correct_graph)
    nx.draw(correct_graph, pos, with_labels=True, ax=axs[0])

    nx.draw_networkx_edge_labels(correct_graph, pos, edge_labels=nx.get_edge_attributes(correct_graph, 'relation'), ax=axs[0])
    axs[0].set_title('Correct graph')
    
    pos = nx.nx_agraph.graphviz_layout(predicted_graph)
    nx.draw(predicted_graph, pos, with_labels=True, ax=axs[1])
    
    nx.draw_networkx_edge_labels(predicted_graph, pos, edge_labels=nx.get_edge_attributes(predicted_graph, 'relation'), ax=axs[1])
    axs[1].set_title('Predicted graph')
    
    
    fig.suptitle(sentence)
    plt.savefig(path)
    
    plt.close()
    
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sentences_path', type=str)
    parser.add_argument('--path_truth', type=str)
    parser.add_argument('--path_pred', type=str)
    
    
    args = parser.parse_args()
    
    with open(args.sentences_path, 'r') as f:
        sentences = f.read().splitlines()
    
    with open(args.path_truth, 'r') as f:
        all_correct_graphs = f.read().splitlines()
    with open(args.path_pred, 'r') as f:
        all_predicted_graphs = f.read().splitlines()
        
    for i, (sentence, correct_graph_line, predicted_graph_line) in tqdm(enumerate(zip(sentences, all_correct_graphs, all_predicted_graphs))):
        try:
            correct_graph = parse_graph(correct_graph_line)
            predicted_graph = parse_graph(predicted_graph_line)
            
            visualize(correct_graph, predicted_graph, sentence, os.path.join('graphs', str(i) + '.png'))
        except:
            print('Error in sentence: ', sentence)
            continue