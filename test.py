import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from collections import deque

# Import your ID3 implementation
import CAMPUS_SECTION_SRN_Lab3 as id3


def evaluate_model(X, y, tree, dataset_name):
    """
    Evaluate the decision tree on given dataset
    """
    y_pred = []
    for row in X:
        y_pred.append(predict(tree, row))

    # Convert to numpy arrays
    y_true = np.array(y, dtype=int)
    y_pred = np.array(y_pred, dtype=int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    # Tree complexity
    depth = get_tree_depth(tree)
    size = get_tree_size(tree)

    print(f"\n--- Results for {dataset_name} ---")
    print(f"Accuracy   : {acc:.4f}")
    print(f"Precision  : {precision:.4f}")
    print(f"Recall     : {recall:.4f}")
    print(f"F1-Score   : {f1:.4f}")
    print(f"Tree Depth : {depth}")
    print(f"Tree Size  : {size}")
    print("-------------------------------\n")


def predict(tree, row):
    """
    Traverse the decision tree to make prediction for a row
    """
    while isinstance(tree, dict):
        attr = next(iter(tree))
        value = row[attr]
        if value in tree[attr]:
            tree = tree[attr][value]
        else:
            # fallback if unseen attribute value
            return 0
    return tree


def get_tree_depth(tree):
    """
    Compute maximum depth of the tree
    """
    if not isinstance(tree, dict):
        return 1
    else:
        return 1 + max(get_tree_depth(subtree) for subtree in tree[next(iter(tree))].values())


def get_tree_size(tree):
    """
    Compute total number of nodes in the tree
    """
    if not isinstance(tree, dict):
        return 1
    else:
        size = 1
        for subtree in tree[next(iter(tree))].values():
            size += get_tree_size(subtree)
        return size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ID", type=str, required=True, help="Your file ID (CAMPUS_SECTION_SRN_Lab3)")
    parser.add_argument("--data", type=str, required=True, help="Dataset file (mushroom.csv, tictactoe.csv, nursery.csv)")
    parser.add_argument("--print-tree", action="store_true", help="Print the trained decision tree")
    args = parser.parse_args()

    # Load dataset
    data = pd.read_csv(args.data)
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the tree using ID3
    dataset_train = np.column_stack((X_train, y_train))
    tree = id3.id3(dataset_train)

    # Evaluate
    evaluate_model(X_test, y_test, tree, args.data)

    # Print tree if requested
    if args.print_tree:
        print("\nDecision Tree:")
        print(tree)


if __name__ == "__main__":
    main()
