def number_nodes(components_list, components_nodes_list, connections_list):
    """This function modifies the number attribute in the nodes of the componentes nodes lists
    """

    # Generate the connection matrix
    connection_matrix = []
    for connection in connections_list:
        connection_matrix.append(connection.descriptor)

    node_counter = 0

    for i, component in enumerate(components_list):

        for j, node in enumerate(components_nodes_list[i]):

            # If the node is the root of the component check if it is connected to a node tha has
            # already been numbered
            if j == 0:
                # Check if node is part of a connection
                node_identifier = component.identifier + "-ROOT"
                connections = check_connections(connections_list, components_list, node_identifier)

                # Run through the connections and selects connected nodes
                connected_nodes = []
                for connection in connections:

                    connected_comp_index = components_list.index(connection[0])
                    connected_comp_nodes = components_nodes_list[connected_comp_index]

                    if connection[1] == "ROOT":
                        connected_node = connected_comp_nodes[0]

                    elif connection[1] == "TIP":
                        last_index = len(connected_comp_nodes) - 1
                        connected_node = connected_comp_nodes[last_index]

                    connected_nodes.append(connected_node)

                # Run through the connected nodes and checks if one of the is numbered
                node_number = None
                for connected_node in connected_nodes:
                    if connected_node.number is not None:
                        node_number = connected_node.number

                if node_number is not None:
                    node.number = node_number
                else:
                    node.number = node_counter
                    node_counter += 1

            # If the node is the tip of the component check if it is connected to a node tha has
            # already been numbered, same as the root
            if j == len(components_nodes_list[i]) - 1:
                # Check if node is part of a connection
                node_identifier = component.identifier + "-TIP"
                connections = check_connections(connections_list, components_list, node_identifier)

                # Run through the connections and selects connected nodes
                connected_nodes = []
                for connection in connections:

                    connected_comp_index = components_list.index(connection[0])
                    connected_comp_nodes = components_nodes_list[connected_comp_index]

                    if connection[1] == "ROOT":
                        connected_node = connected_comp_nodes[0]

                    elif connection[1] == "TIP":
                        last_index = len(connected_comp_nodes) - 1
                        connected_node = connected_comp_nodes[last_index]

                    connected_nodes.append(connected_node)

                # Run through the connected nodes and checks if one of the is numbered
                node_number = None
                for connected_node in connected_nodes:
                    if connected_node.number is not None:
                        node_number = connected_node.number

                if node_number is not None:
                    node.number = node_number
                else:
                    node.number = node_counter
                    node_counter += 1

            # If the node is not at the ROOT or the TIP give it the next avaiable number:
            else:
                node.number = node_counter
                node_counter += 1

# ==================================================================================================


def check_connections(connections_list, components_list, node_identifier):
    """This function returns all the connections to a node
    """
    connections = []

    # Create a list with all the connection identifiers
    connection_matrix = []
    for connection in connections_list:
        connection_matrix.append(connection.descriptor)

    # Run through the connections
    for i, connection in enumerate(connection_matrix):

        # Finds a connection that contains the node identifier
        if node_identifier in connection:

            # Connections contains two components, index tells if it is the first or second
            # component
            index = connection.index(node_identifier)
            connected_component_index = abs(index - 1)
            connected_component = connections_list[i][connected_component_index * 2]
            connected_node = connections_list[i][connected_component_index * 2 + 1]

            connections.append([connected_component, connected_node])

    return connections

# ==================================================================================================








