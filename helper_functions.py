import numpy as np

def squaredist_full(x, y, z):
        return np.sqrt((x[1:] - x[:-1]) ** 2+\
                       (y[1:] - y[:-1]) ** 2+\
                       (z[1:] - z[:-1]) ** 2)

def dotproduct_traces(data):
    """
    Computes the dotproduct for a trajectory (x,y)
    """
    dotproducts = []
    for trace in data:
        vecs = trace[1:]-trace[:-1]
        dots = np.dot(vecs[:-1], vecs[1:].T).diagonal()
        dots = np.append([0, 0], dots)
        dotproducts.append(dots.reshape(-1,1))
    return np.array(dotproducts, dtype=object)


def steplength_traces(data):
    """
    Computes the steplength in 2. norm for a trajectory (x,y)
    """
    steplengths = []
    for trace in data:
        if trace.shape[1]==2:
            x, y = trace[:,0], trace[:,1]
            z = np.zeros_like(x)
        if trace.shape[1]==3:
            x, y, z = trace[:,0], trace[:,1], trace[:,2]
        sl = squaredist_full(x, y, z)
        sl = np.append(0, sl)
        steplengths.append(sl.reshape(-1,1))
    return np.array(steplengths, dtype=object)

def origin_distance(data):
    """
    Computes the distance from the origin for a trajectory (x,y)
    """

    distances = []
    for trace in data:
        if trace.shape[1]==2:
            x, y = trace[:,0]-trace[0,0], trace[:,1]-trace[0,1]
            z = np.zeros_like(x)
        if trace.shape[1]==3:
            x, y, z = trace[:,0]-trace[0,0], trace[:,1]-trace[0,1], trace[:,2]-trace[0,2]

        dist = np.sqrt(x**2 + y**2 + z**2)

        distances.append(dist.reshape(-1,1))
    return np.array(distances, dtype=object)

def euclidian_coordinates_to_polar_or_sphere(data):
    """
    Converts euclidian coordinates to polar or spherical coordinates
    """
    polar = []
    for trace in data:
        if trace.shape[1]==2:
            x, y = trace[:,0], trace[:,1]
            z = np.zeros_like(x)
            r = np.sqrt(x**2 + y**2)
            theta = np.arctan2(y, x)
            polar.append(np.vstack([r, theta]).T)

        if trace.shape[1]==3:
            x, y, z = trace[:,0], trace[:,1], trace[:,2]
            r = np.sqrt(x**2 + y**2 + z**2)
            theta = np.arctan2(y, x)
            phi = np.arccos(z/r)
            polar.append(np.vstack([r, theta, phi]).T)
        
    return np.array(polar, dtype=object)

def add_features(X, features_list: list):
    dim = X[0].shape[1]
    out = X
    if 'SL' in features_list:
        steplengths = steplength_traces(X)
        out = [np.hstack([out[i], steplengths[i]]) for i in range(len(X))]
    if 'DP' in features_list:
        dotproduct = dotproduct_traces(X)
        out = [np.hstack([out[i], dotproduct[i]]) for i in range(len(X))]
    if 'origin_distance' in features_list:
        origin = origin_distance(X)
        out = [np.hstack([out[i], origin[i]]) for i in range(len(X))]
    if 'polar' in features_list:
        polar = euclidian_coordinates_to_polar_or_sphere(X)
        out = [np.hstack([out[i], polar[i]]) for i in range(len(X))]
    if 'XYZ' not in features_list:
        out = [x[:,dim:] for x in out]

        
    return out