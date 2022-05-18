def get_class_mapping(C, C_dash):
    all_class = list(set(C).union(C_dash))
    all_class.sort()
    class_mapping = {content: i for i, content in enumerate(all_class)}
    return class_mapping


def get_domain_mapping(src_domains, trgt_domains):
    src_dm = [k for k in src_domains]
    trgt_dm = [k for k in trgt_domains]

    src_dm.sort()
    trgt_dm.sort()

    dom_mapping = {j: i for i, j in enumerate(src_dm)}

    for i, j in enumerate(trgt_dm):
        dom_mapping[j] = i + len(src_dm)

    return dom_mapping
