"""
Boilerplate for ID3 implementation.
Fill these functions in your lab task.
"""

def get_entropy_of_dataset(dataset):
    """
    Calculate entropy of the dataset.
    Input: pandas DataFrame (last column = target class)
    Output: entropy (float)
    """
    pass


def get_avg_info_of_attribute(dataset, attribute):
    """
    Calculate average information (expected entropy) for given attribute.
    Input: dataset (pandas DataFrame), attribute (str)
    Output: average entropy (float)
    """
    pass


def get_information_gain(dataset, attribute):
    """
    Calculate information gain of a given attribute.
    Input: dataset (pandas DataFrame), attribute (str)
    Output: information gain (float)
    """
    pass


def get_selected_attribute(dataset):
    """
    Select attribute with maximum information gain.
    Input: dataset (pandas DataFrame)
    Output: best attribute name (str), dictionary of info gains
    """
    pass
