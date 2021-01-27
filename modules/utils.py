

def get_hidden_layer_sizes(start_size, end_size, n_hidden_layers):
    """
    It can handle both increasing & decreasing sizes automatically
    """
    sizes = []
    diff = (start_size - end_size) / (n_hidden_layers + 1)

    for idx in range(n_hidden_layers):
        sizes.append(int(start_size - (diff * (idx + 1))))
    return sizes