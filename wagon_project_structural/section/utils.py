import numpy as np
def beams_sample(n_attrs,attrs_ranges,cls_beam,n_beams = 100):
    """

    :param n_attrs:
    :param attrs_ranges:
    for n_attrs = 4
    attrs_ranges = np.array(
            [
                [0.1,0.9],
                [0.1,1.3],
                [0.001,0.015],
                [0.001,0.01]

            ]
        )
    :param n_beams:
    :return:
    """
    sample = np.random.rand(n_beams,n_attrs)
    loc = attrs_ranges.min(axis=1)
    scale = attrs_ranges[:,1]-attrs_ranges[:,0]
    return list(map(lambda args:cls_beam(*args),(np.multiply(sample,scale)+loc)))