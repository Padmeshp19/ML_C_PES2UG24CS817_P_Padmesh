import pandas as pd
import numpy as np
import argparse

# ------------------------------------------------------------
# Node class for decision tree
# ------------------------------------------------------------
class Node:
    def __init__(self, feature=None, label=None):
        self.feature = feature  # splitting attribute
        self.label = label      # class label if leaf node
        self.children = {}      # dictionary mapping feature values to child nodes

# ------------------------------------------------------------
# Entropy function
# ------------------------------------------------------------
def entropy(col):
    elements, counts = np.unique(col, return_counts=True)
    probabilities = counts / counts.sum()
    return -np.sum(probabilities * np.log2(probabilities))

# ------------------------------------------------------------
# Information Gain function
# ------------------------------------------------------------
def info_gain(data, attribute, target):
    total_entropy = entropy(data[target])
    values, counts = np.unique(data[attribute], return_counts=True)
    weighted_entropy = 0
    for v, c in zip(values, counts):
        subset = data[data[attribute] == v]
        weighted_entropy += (c / len(data)) * entropy(subset[target])
    return total_entropy - weighted_entropy

# ------------------------------------------------------------
# Majority class function
# ------------------------------------------------------------
def majority_class(col):
    return col.mode()[0]

# ------------------------------------------------------------
# ID3 Algorithm
# ------------------------------------------------------------
def id3(data, target_attribute=None):
    if target_attribute is None:
        target_attribute = data.columns[-1]  # assume last column is target

    # Case 1: All rows have same class → return leaf node
    if len(np.unique(data[target_attribute])) == 1:
        return Node(label=np.unique(data[target_attribute])[0])

    # Case 2: No features left → return leaf node with majority class
    if len(data.columns) == 1:
        return Node(label=majority_class(data[target_attribute]))

    # Select best attribute based on information gain
    attributes = [col for col in data.columns if col != target_attribute]
    gains = [info_gain(data, a, target_attribute) for a in attributes]
    best_attr = attributes[np.argmax(gains)]

    # Create node for best attribute
    node = Node(feature=best_attr)

    # Branch for each value of the best attribute
    for val in np.unique(data[best_attr]):
        subset = data[data[best_attr] == val].drop(columns=[best_attr])
        node.children[val] = id3(subset, target_attribute)

    return node

# ------------------------------------------------------------
# Prediction function
# ------------------------------------------------------------
def predict(node, test_data, default=None):
    predictions = []

    for _, row in test_data.iterrows():
        curr = node
        while curr.label is None:
            val = row[curr.feature]
            if val in curr.children:
                curr = curr.children[val]
            else:
                # fallback to majority class
                curr = Node(label=default)
        predictions.append(curr.label)
    return predictions

# ------------------------------------------------------------
# Tree Depth
# ------------------------------------------------------------
def tree_depth(node):
    if node.label is not None:
        return 0
    return 1 + max(tree_depth(child) for child in node.children.values())

# ------------------------------------------------------------
# Tree Size
# ------------------------------------------------------------
def tree_size(node):
    if node.label is not None:
        return 1
    return 1 + sum(tree_size(child) for child in node.children.values())

# ------------------------------------------------------------
# Print tree (matches boilerplate)
# ------------------------------------------------------------
def print_tree(node, depth=0):
    if node.label is not None:
        print("  " * depth + f"Leaf: {node.label}")
    else:
        print("  " * depth + f"[Feature: {node.feature}]")
        for val, child in node.children.items():
            print("  " * (depth + 1) + f"Value={val}:")
            print_tree(child, depth + 2)

# ------------------------------------------------------------
# Main section for standalone running
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ID3 Decision Tree Script")
    parser.add_argument("--ID", type=str, help="Your Lab ID", required=False)
    parser.add_argument("--data", type=str, help="Path to dataset CSV", required=True)
    parser.add_argument("--print-tree", action="store_true", help="Print full decision tree")
    args = parser.parse_args()

    # Load dataset
    try:
        data = pd.read_csv(args.data)
    except FileNotFoundError:
        print(f"Error: File '{args.data}' not found.")
        exit()

    # Display Lab ID if provided
    if args.ID:
        print("Lab ID:", args.ID)

    # Target column (assume last column)
    target = data.columns[-1]

    # Default class for unseen values
    default_class = majority_class(data[target])

    # Build tree
    tree = id3(data, target_attribute=target)

    # Print tree if requested
    if args.print_tree:
        print("\n--- Decision Tree ---")
        print_tree(tree)

    # Predictions on training data
    predictions = predict(tree, data, default=default_class)
    accuracy = np.mean(predictions == data[target])

    # Print comparative analysis report
    print("\n--- Comparative Analysis Report ---")
    print(f"accuracy: {accuracy:.4f}")
    print(f"depth: {tree_depth(tree)}")
    print(f"size: {tree_size(tree)}")
    print("\nSample predictions:", predictions[:10])

if __name__ == "__main__":
    main()
