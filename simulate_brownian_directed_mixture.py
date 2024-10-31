# %%
import numpy as np
import matplotlib.pyplot as plt
import pickle

initial_state = [0,0,0] 
beta_set = None

D = 0.01  # diffusion coefficient
dt = 1  # time step
R = 15  # contribution of directed motion
Q1 = 10  # contribution of loc error
sig = np.sqrt(D * dt) / Q1  # noise standard deviation
L = 999  # number of steps
N = 10000  # number of trials
dim = 3  # dimension of the motion

v = 0.1
vs = np.random.uniform(0, v, size=N)
print(vs)

plt.figure(figsize=(5, 4))
plt.hist(vs, bins=100)
plt.show()

tracks = []
for v in vs:
    # Simulate Brownian motion mixed with directed motion
    x0, y0, z0 = initial_state[0], initial_state[1], initial_state[2]

    if beta_set is not None: 
        theta, phi = beta_set
    else:
        theta_set, phi_set = None, None

    if theta_set is None:
        theta = np.random.uniform(0, np.pi)
    if phi_set is None:
        phi = np.random.uniform(0, 2 * np.pi)

    dx = v * dt * np.sin(phi)*np.cos(theta)
    dy = v * dt * np.sin(phi)*np.sin(theta)
    dz = v * dt * np.cos(phi)


    xsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=L) + dx
    ysteps = np.random.normal(0, np.sqrt(2 * D * dt), size=L) + dy
    zsteps = np.random.normal(0, np.sqrt(2 * D * dt), size=L) + dz

    x, y, z = (
        np.concatenate([[x0], np.cumsum(xsteps)+x0]),
        np.concatenate([[y0], np.cumsum(ysteps)+y0]),
        np.concatenate([[z0], np.cumsum(zsteps)+z0])
    )

    x_noisy, y_noisy, z_noisy = (
        x + np.random.normal(0, sig, size=x.shape),
        y + np.random.normal(0, sig, size=y.shape),
        z + np.random.normal(0, sig, size=z.shape)
    )

    track = np.vstack((x_noisy, y_noisy, z_noisy)).T
    tracks.append(track)

pickle.dump(tracks, open("data/directed_tracks/tracks.pkl", "wb"))
pickle.dump(vs, open("data/directed_tracks/speeds.pkl", "wb"))


# %%

i = np.random.choice(np.arange(len(tracks)), size=1)[0]
track = tracks[i]

SL = np.sqrt(np.sum(np.diff(track, axis=0)**2, axis=1))
xdisplacement = track[1:, 0] - track[:-1, 0]
ydisplacement = track[1:, 1] - track[:-1, 1]
zdisplacement = track[1:, 2] - track[:-1, 2]

plt.figure(figsize=(5, 4))
plt.hist(SL, bins=30)

plt.figure(figsize=(5, 4))
plt.plot(track[:, 0], track[:, 1])
# equal x and y aspec ratio
plt.gca().set_aspect('equal', adjustable='box')

plt.figure(figsize=(5, 4))
plt.hist(xdisplacement, bins=30)
plt.hist(ydisplacement, bins=30)
plt.hist(zdisplacement, bins=30)

