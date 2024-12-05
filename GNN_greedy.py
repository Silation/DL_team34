import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import os

def build_greedy_tour(node_file, edge_file):
    """
    저장된 네트워크 데이터를 기반으로 Greedy 알고리즘을 사용해 최적 경로 생성
    """
    # 데이터 불러오기
    nodes = pd.read_csv(node_file)
    edges = pd.read_csv(edge_file)

    # NetworkX 그래프 초기화
    graph = nx.Graph()
    original_edges = []  # 모든 엣지 저장

    # 노드 추가
    for _, row in nodes.iterrows():
        graph.add_node(row["node_id"], pos=(row["x"], row["y"]))

    # 최적 경로 엣지 추가
    for _, row in edges.iterrows():
        original_edges.append((row["from_node"], row["to_node"]))
        if row["is_in_tour"] == 1:
            graph.add_edge(row["from_node"], row["to_node"], weight=row["distance"])

    # Greedy 알고리즘으로 경로 연결
    while nx.number_connected_components(graph) > 1:  # 파편 연결
        components = list(nx.connected_components(graph))
        pairs = []

        # 파편 간 가장 가까운 노드 쌍 찾기
        for i, comp1 in enumerate(components):
            for j, comp2 in enumerate(components):
                if i < j:
                    for node1 in comp1:
                        if graph.degree[node1] >= 2:  # 이미 2개의 엣지를 가진 노드 제외
                            continue
                        for node2 in comp2:
                            if graph.degree[node2] >= 2:  # 이미 2개의 엣지를 가진 노드 제외
                                continue
                            pos1 = graph.nodes[node1]["pos"]
                            pos2 = graph.nodes[node2]["pos"]
                            distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
                            pairs.append((node1, node2, distance))

        # 연결 가능한 노드 쌍이 없는 경우 종료
        if not pairs:
            print("No more valid pairs to connect.")
            break

        # 가장 가까운 노드 쌍 연결
        pairs.sort(key=lambda x: x[2])  # 거리 기준 정렬
        from_node, to_node, _ = pairs[0]
        graph.add_edge(from_node, to_node, weight=_)

        # 사이클 체크
        try:
            cycle = nx.find_cycle(graph)
            # 사이클이 있다면 가장 긴 엣지 제거
            if cycle:
                max_edge = max(cycle, key=lambda x: graph[x[0]][x[1]]["weight"])
                graph.remove_edge(max_edge[0], max_edge[1])
        except nx.NetworkXNoCycle:
            pass  # 사이클이 없으면 무시

    # 엣지 수가 하나인 노드 찾기
    degree_one_nodes = [node for node in graph.nodes() if graph.degree[node] == 1]

    # 마지막 두 노드를 연결하여 하나의 사이클 생성
    if len(degree_one_nodes) == 2:
        node1, node2 = degree_one_nodes
        pos1 = graph.nodes[node1]["pos"]
        pos2 = graph.nodes[node2]["pos"]
        distance = ((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)**0.5
        graph.add_edge(node1, node2, weight=distance)

    # 결과 저장
    final_edges = list(graph.edges(data=True))
    return graph, final_edges, original_edges



def visualize_greedy_tour(graph, original_edges, save_path):
    """
    생성된 경로를 시각화하여 그림으로 저장 (전체 그래프 엣지 포함, 노드 원 추가)
    """
    pos = nx.get_node_attributes(graph, 'pos')  # 노드의 위치 정보

    plt.figure(figsize=(10, 10))

    # 모든 엣지 (회색으로 표시)
    full_graph = nx.Graph()
    full_graph.add_edges_from(original_edges)
    nx.draw(
        full_graph, pos,
        with_labels=False,
        node_size=0,
        edge_color="gray",
        alpha=0.5,
        width=0.1
    )

    # 최적 경로 강조 (빨간색)
    nx.draw_networkx_edges(
        graph, pos,
        edgelist=graph.edges(data=True),
        edge_color="red",
        width=1.5
    )

    # 노드 원 추가
    nx.draw_networkx_nodes(
        graph, pos,
        node_size=300,
        alpha=0.5
    )

    # 노드 번호를 정수로 표시
    nx.draw_networkx_labels(
        graph, pos,
        labels={node: int(node) for node in graph.nodes()},
        font_size=8,
        font_color="black"
    )

    plt.title("Greedy TSP Solution")
    plt.savefig(save_path)
    print(f"Greedy tour graph saved to {save_path}")
    plt.close()


    

# 테스트 데이터의 최적 경로 생성 및 저장
def process_saved_data_for_greedy_tour(saved_folder="network_info", save_graph_dir="greedy_graphs"):
    """
    저장된 데이터를 기반으로 Greedy 알고리즘을 실행하고 시각화 결과를 저장
    """
    if not os.path.exists(save_graph_dir):
        os.makedirs(save_graph_dir)

    node_files = [f for f in os.listdir(saved_folder) if f.endswith("_nodes.csv")]
    edge_files = [f for f in os.listdir(saved_folder) if f.endswith("_edges.csv")]

    # 각 테스트 데이터에 대해 경로 생성
    for node_file, edge_file in zip(sorted(node_files), sorted(edge_files)):
        node_path = os.path.join(saved_folder, node_file)
        edge_path = os.path.join(saved_folder, edge_file)

        # Greedy 경로 생성
        graph, final_edges, original_edges = build_greedy_tour(node_path, edge_path)

        # 시각화 저장
        test_id = node_file.split("_nodes.csv")[0]  # test ID 추출
        save_path = os.path.join(save_graph_dir, f"{test_id}_greedy_tour.png")
        visualize_greedy_tour(graph, original_edges, save_path)
        
# 실행
process_saved_data_for_greedy_tour()
