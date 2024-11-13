import random
import matplotlib.pyplot as plt

def generate_random_nodes(num_nodes, width=3000, height=3000):
    """
    Generates nodes with random positions within a 100x100 grid.
    
    Parameters:
        num_nodes (int): Number of nodes to generate.
        width (int): Width of the grid.
        height (int): Height of the grid.
    
    Returns:
        dict: A dictionary with nodes as keys and their (x, y) positions as values.
    """
    positions = [(random.randint(0, width), random.randint(0, height)) for _ in range(num_nodes)]
    return positions

def plot_nodes(positions, width=100, height=100):
    """
    Plots nodes on a 100x100 grid using matplotlib.
    
    Parameters:
        positions (dict): A dictionary with nodes as keys and their (x, y) positions as values.
        width (int): Width of the grid.
        height (int): Height of the grid.
    """
    plt.figure(figsize=(8, 8))
    plt.xlim(0, width)
    plt.ylim(0, height)
    
    # Plot each node as a point
    for (x, y) in positions:
        plt.scatter(x, y, c='blue')
        #plt.text(x, y, str(node), fontsize=9, ha='right')  # Display node label next to the point
    
    plt.title("Random Nodes in a 100x100 Grid")
    plt.xlabel("Width")
    plt.ylabel("Height")
    plt.grid(True)
    plt.show()

# Example Usage
# Generate 10 nodes with random positions in a 100x100 grid and plot them
"""
nodes = generate_random_nodes(10)
print(nodes)
plot_nodes(nodes)
"""