import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from numpy.linalg import inv
from numpy.linalg import svd

def write_image(vector, dim, filename):
    vector = ((vector+1.0)/2.0)*255.0
    vector = np.reshape(vector, dim)
    p = vector.astype("uint8")
    p = cv2.resize(p, (100, 100))
    count = 0
    cv2.imwrite("images/" + filename, p)
    
def main():
    with open("x_train.p", "rb") as f:
        x_train = np.array(pickle.load(f))

    with open("y_train.p", "rb") as f:
        y_train = np.array(pickle.load(f))

    with open("x_test.p", "rb") as f:
        x_test = np.array(pickle.load(f))

    with open("y_test.p", "rb") as f:
        y_test = np.array(pickle.load(f))

    # Get image dimensions
    N_train, m_train, n_train, k_train = x_train.shape
    M_train = m_train * n_train * k_train
    N_test, m_test, n_test, k_test = x_test.shape
    M_test = m_test * n_test * k_test

    # Flatten images
    x_train = x_train.reshape(N_train, M_train)
    y_train = y_train.reshape(N_train, M_train)
    x_test = x_test.reshape(N_test, M_test)
    y_test = y_test.reshape(N_test, M_test)

    # Standardize the data
    x_train = (x_train/255)*2 - 1
    y_train = (y_train/255)*2 - 1
    x_test = (x_test/255)*2 - 1
    y_test = (y_test/255)*2 - 1

    # Compute the covariance matrices
    Sxy = np.dot(x_train.T - np.average(x_train, axis=1), (y_train.T - np.average(y_train, axis=1)).T)
    Sxx = np.dot(x_train.T - np.average(x_train, axis=1), (x_train.T - np.average(x_train, axis=1)).T)
    Syy = np.dot(y_train.T - np.average(y_train, axis=1), (y_train.T - np.average(y_train, axis=1)).T)

    # SVD
    alpha = 0.00001
    U, S, V = svd(np.dot(inv(sqrtm(Sxx + alpha*np.identity(M_train))), np.dot(Sxy, inv(sqrtm(Syy + alpha*np.identity(M_train))))))

    # Plot singular value spectrum
    plt.plot(np.arange(0, M_train), S)
    plt.title("Singular Value Spectrum")
    plt.xlabel("Signular value rank")
    plt.ylabel("Sigular value")
    plt.show()

    # Build first eigenface
    p = U[:, 0] # Get first left eigenvector
    eigenface = np.dot(x_train[0], p) * p + np.mean(x_train[0]) # Project on eigenvector
    write_image(eigenface, (m_train, n_train, k_train), "eigenface.png")

    # Learn mapping using ridge regression
    ks = np.arange(50, 700, 50)
    test_error = list()
    for k in ks:
        Pk = U[:, :k] # Get the appropriate eigenspace
        Xp = np.dot(x_train, Pk) # Compute the projection in the eigenspace
        # Learn the model
        w = np.dot(np.dot(inv(np.dot(Xp.T, Xp) + alpha*np.identity(k)), Xp.T), y_train)

        # Compute the validation error
        error = np.linalg.norm(np.dot(np.dot(x_test, Pk), w) - y_test, 2)
        test_error.append(error)

    # Plot the test error
    plt.plot(ks, test_error)
    plt.xlabel("k")
    plt.ylabel("Squared Euclidean Test Error")
    plt.title("Model Training")
    plt.show()

    # Predict 4 images
    for i in range(0, 4):
        write_image(x_test[i], (m_test, n_test, k_test), "mooney" + str(i) + ".png")
        write_image(np.dot(np.dot(x_test[i], Pk), w), (m_test, n_test, k_test), "predicted" + str(i) + ".png")
        write_image(y_test[i], (m_test, n_test, k_test), "ground_truth" + str(i) + ".png")

if __name__ == "__main__":
    main()
