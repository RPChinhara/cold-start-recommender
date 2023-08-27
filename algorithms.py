def item_attribute_generation_matrix(i_old, i_new, item_attributes: dict):
    '''
    i_old is a set containing Old Items\n
    i_new is a set containg New Items\n
    item_attributes is a dictionary containg Attributes\n
    key: item_id, value: attribute
    '''

    A_set = set()
    IA_matrix = list()
    
    for attributes in item_attributes.values():
        A_set.update(attributes)

    for item_id in i_old:
        attributes = item_attributes.get(item_id, [])
        attribute_vector = [1 if attr in attributes else 0 for attr in A_set]
        IA_matrix.append(attribute_vector)

    for item_id in i_new:
        attributes = item_attributes.get(item_id, [])
        attribute_vector = [1 if attr in attributes else 0 for attr in A_set]
        IA_matrix.append(attribute_vector)

    return IA_matrix